from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cotracker_utils import load_video_frames, prepare_davis_frames  # noqa: E402


DEFAULT_LINGBOT_ROOT = Path("/data/project-vilab/jaeseok/lingbot-world")
DEFAULT_DAVIS_ROOT = Path("/data/project-vilab/jaeseok/davis/DAVIS/JPEGImages")
DEFAULT_CKPT_DIR = DEFAULT_LINGBOT_ROOT / "lingbot-world-base-cam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LingBot Wan2.1 VAE full decode with stateful streaming "
            "chunk decode on a DAVIS frame sequence."
        )
    )
    parser.add_argument("--lingbot_root", type=Path, default=DEFAULT_LINGBOT_ROOT)
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--video_path", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--sequence", type=str, default="blackswan")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "1080p"])
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_stateless_control", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def resolve_video_path(path: Path, resolution: str, sequence: str) -> Path:
    if path.is_dir() and (path / resolution / sequence).is_dir():
        return path / resolution / sequence
    if path.is_dir() and (path / sequence).is_dir():
        return path / sequence
    return path


def frames_to_vae_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T, H, W, 3), got {frames.shape}")
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    tensor = tensor.div_(127.5).sub_(1.0)
    return tensor.to(device)


def vae_tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    arr = video.detach().float().cpu().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).round().byte()
    return arr.permute(1, 2, 3, 0).numpy()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return math.inf
    # Values are in [-1, 1], so the peak-to-peak range is 2.
    return 10.0 * math.log10(4.0 / mse)


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": 0.0,
            "reference_frames": float(reference.shape[1]),
            "candidate_frames": float(candidate.shape[1]),
        }
    diff = (reference - candidate).abs().float()
    mse = torch.mean((reference.float() - candidate.float()) ** 2).item()
    return {
        "shape_match": 1.0,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "p99_abs": torch.quantile(diff.flatten(), 0.99).item(),
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
    }


def scale_latents_for_decode(vae, z: torch.Tensor) -> torch.Tensor:
    if isinstance(vae.scale[0], torch.Tensor):
        mean = vae.scale[0].view(1, vae.model.z_dim, 1, 1, 1)
        inv_std = vae.scale[1].view(1, vae.model.z_dim, 1, 1, 1)
        return z / inv_std + mean
    return z / vae.scale[1] + vae.scale[0]


def decode_stream_chunks(
    vae,
    latent: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    model = vae.model
    model.clear_cache()
    chunks: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    frame_cursor = 0
    enabled = device.type == "cuda"

    try:
        autocast_kwargs = {"device_type": device.type, "enabled": enabled}
        if enabled:
            autocast_kwargs["dtype"] = vae.dtype
        with torch.amp.autocast(**autocast_kwargs):
            for start in range(0, latent.shape[1], chunk_size):
                z_chunk = latent[:, start : start + chunk_size].unsqueeze(0)
                z_chunk = scale_latents_for_decode(vae, z_chunk)
                x = model.conv2(z_chunk)

                per_latent_outputs = []
                for latent_idx in range(x.shape[2]):
                    model._conv_idx = [0]
                    out_i = model.decoder(
                        x[:, :, latent_idx : latent_idx + 1],
                        feat_cache=model._feat_map,
                        feat_idx=model._conv_idx,
                    )
                    per_latent_outputs.append(out_i)

                chunk_video = torch.cat(per_latent_outputs, dim=2).float().clamp_(-1, 1)
                chunk_frames = int(chunk_video.shape[2])
                spans.append((frame_cursor, frame_cursor + chunk_frames))
                frame_cursor += chunk_frames
                chunks.append(chunk_video)
    finally:
        model.clear_cache()

    return torch.cat(chunks, dim=2).squeeze(0), spans


def decode_stateless_chunks(vae, latent: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, latent.shape[1], chunk_size):
        chunks.append(vae.decode([latent[:, start : start + chunk_size]])[0])
    return torch.cat(chunks, dim=1)


def boundary_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    spans: list[tuple[int, int]],
) -> list[dict[str, float]]:
    stats = []
    for _, end in spans[:-1]:
        left = max(0, end - 2)
        right = min(reference.shape[1], end + 2)
        if right <= left or candidate.shape[1] < right:
            continue
        diff = (reference[:, left:right] - candidate[:, left:right]).abs().float()
        stats.append(
            {
                "boundary_after_frame": float(end - 1),
                "window_start": float(left),
                "window_end": float(right),
                "max_abs": diff.max().item(),
                "mean_abs": diff.mean().item(),
            }
        )
    return stats


def save_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")


def load_lingbot_vae_class(lingbot_root: Path):
    vae_path = lingbot_root / "wan" / "modules" / "vae2_1.py"
    if not vae_path.is_file():
        raise FileNotFoundError(f"LingBot VAE module not found: {vae_path}")
    spec = importlib.util.spec_from_file_location("lingbot_vae2_1", vae_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load import spec for: {vae_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_1_VAE


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    lingbot_root = args.lingbot_root.resolve()
    Wan2_1_VAE = load_lingbot_vae_class(lingbot_root)

    video_path = resolve_video_path(args.video_path, args.resolution, args.sequence)
    frames = load_video_frames(video_path, target_hw=(args.height, args.width))
    frames, selected_indices = prepare_davis_frames(frames, args.frames)
    if frames.shape[0] != args.frames:
        raise ValueError(
            f"Loaded {frames.shape[0]} frames, but --frames={args.frames}. "
            "Use a longer sequence or lower --frames."
        )

    vae_pth = args.ckpt_dir / "Wan2.1_VAE.pth"
    if not vae_pth.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_pth}")

    print(f"[vae_stream_eval] video_path={video_path}")
    print(f"[vae_stream_eval] selected_indices={selected_indices.tolist()}")
    print(f"[vae_stream_eval] frames={frames.shape}, device={device}")
    print(f"[vae_stream_eval] vae_pth={vae_pth}")

    video = frames_to_vae_tensor(frames, device)
    vae = Wan2_1_VAE(vae_pth=str(vae_pth), device=device)

    sync(device)
    t0 = time.perf_counter()
    latent = vae.encode([video])[0]
    sync(device)
    encode_s = time.perf_counter() - t0

    sync(device)
    t0 = time.perf_counter()
    full = vae.decode([latent])[0]
    sync(device)
    full_decode_s = time.perf_counter() - t0

    sync(device)
    t0 = time.perf_counter()
    stream, spans = decode_stream_chunks(vae, latent, args.chunk_size, device)
    sync(device)
    stream_decode_s = time.perf_counter() - t0

    metrics = {
        "video_path": str(video_path),
        "selected_indices": selected_indices.tolist(),
        "input_shape_t_h_w_c": list(frames.shape),
        "latent_shape_c_t_h_w": list(latent.shape),
        "chunk_size": args.chunk_size,
        "stream_chunk_frame_spans": spans,
        "timing_s": {
            "encode": encode_s,
            "full_decode": full_decode_s,
            "stream_decode": stream_decode_s,
        },
        "stream_vs_full": diff_stats(full, stream),
        "boundary_windows": boundary_stats(full, stream, spans),
    }

    if args.run_stateless_control:
        sync(device)
        t0 = time.perf_counter()
        stateless = decode_stateless_chunks(vae, latent, args.chunk_size)
        sync(device)
        metrics["timing_s"]["stateless_decode"] = time.perf_counter() - t0
        metrics["stateless_vs_full"] = diff_stats(full, stateless)

    print(json.dumps(metrics, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_video(args.output_dir / "input.mp4", frames, args.fps)
        save_video(args.output_dir / "full_decode.mp4", vae_tensor_to_uint8(full), args.fps)
        save_video(args.output_dir / "stream_decode.mp4", vae_tensor_to_uint8(stream), args.fps)
        if full.shape == stream.shape:
            diff = (full - stream).abs().amax(dim=0, keepdim=True).repeat(3, 1, 1, 1)
            diff = diff.clamp(0, 1).mul(2).sub(1)
            save_video(args.output_dir / "absdiff_x2.mp4", vae_tensor_to_uint8(diff), args.fps)
        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
