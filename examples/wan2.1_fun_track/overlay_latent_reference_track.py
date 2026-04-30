import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def _add_repo_root_to_syspath() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _resolve_track_npz(sample_dir: Path, track_npz_path: str) -> Path:
    if track_npz_path:
        resolved = Path(track_npz_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Track npz not found: {resolved}")
        return resolved

    candidates = sorted(sample_dir.glob("transformed_tracks_grid*_survived.npz"))
    if not candidates:
        candidates = sorted(sample_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No npz found under sample dir: {sample_dir}")
    return candidates[0]


def _read_metadata_records(metadata_path: Path) -> List[Dict]:
    suffix = metadata_path.suffix.lower()
    if suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"JSON metadata must be a list: {metadata_path}")
        return rows

    if suffix == ".jsonl":
        rows: List[Dict] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if suffix == ".csv":
        with metadata_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported metadata format: {metadata_path}")


def _resolve_relative_path(path_value: str, search_roots: Sequence[Path]) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return resolved

    attempts: List[Path] = []
    for root in search_roots:
        merged = (root / candidate).resolve()
        attempts.append(merged)
        if merged.exists():
            return merged

    attempt_text = "\n".join(f"  - {p}" for p in attempts)
    raise FileNotFoundError(
        f"Could not resolve relative path: {path_value}\n"
        f"Tried:\n{attempt_text}"
    )


def _resolve_sample_from_metadata(
    metadata_path: Path,
    sample_index: int,
    metadata_data_root: str,
    metadata_track_key: str,
    metadata_latent_key: str,
) -> Tuple[Path, Path, Dict, int]:
    rows = _read_metadata_records(metadata_path)
    if not rows:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    idx = int(sample_index)
    if idx < 0:
        idx += len(rows)
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"sample_index={sample_index} out of range for len={len(rows)}")

    row = rows[idx]
    if not isinstance(row, dict):
        raise ValueError(f"Metadata row must be dict at index {idx}: {type(row).__name__}")

    track_value = row.get(metadata_track_key, "")
    if not track_value:
        raise KeyError(f"Missing `{metadata_track_key}` in metadata row {idx}")
    latent_value = row.get(metadata_latent_key, row.get("file_path", ""))
    if not latent_value:
        raise KeyError(
            f"Missing `{metadata_latent_key}` (or fallback `file_path`) in metadata row {idx}"
        )

    search_roots = [metadata_path.parent, Path.cwd()]
    if metadata_data_root:
        search_roots.insert(0, Path(metadata_data_root).expanduser().resolve())

    track_path = _resolve_relative_path(str(track_value), search_roots)
    latents_path = _resolve_relative_path(str(latent_value), search_roots)
    return latents_path, track_path, row, idx


def _load_tracks(track_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(track_npz))
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        raise KeyError(f"Track file missing tracks key: {track_npz}")

    if "visibility_compressed" in data:
        visibility = data["visibility_compressed"]
    elif "visibility" in data:
        visibility = data["visibility"]
    else:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)

    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_npz}")
    if visibility.ndim != 2:
        raise ValueError(f"Unexpected visibility shape {visibility.shape} in {track_npz}")
    return tracks.astype(np.float32), visibility.astype(np.float32)


def _select_track_indices(
    tracks: np.ndarray,
    visibility: np.ndarray,
    max_points: int,
    sample_mode: str,
    seed: int,
) -> np.ndarray:
    num_points = int(tracks.shape[1])
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    if sample_mode == "uniform":
        idx = np.linspace(0, num_points - 1, max_points)
        return np.unique(np.round(idx).astype(np.int64))

    if sample_mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(num_points, size=max_points, replace=False).astype(np.int64))

    vis_mean = visibility.mean(axis=0) if visibility.size else np.zeros((num_points,), dtype=np.float32)
    if tracks.shape[0] > 1:
        delta = np.diff(tracks, axis=0)  # [T-1, N, 2]
        motion = np.linalg.norm(delta, axis=-1).mean(axis=0)  # [N]
    else:
        motion = np.zeros((num_points,), dtype=np.float32)

    if sample_mode == "top_visibility":
        score = vis_mean
    else:
        # default: top_motion, prioritize visible and dynamic points.
        score = motion * np.clip(vis_mean, 0.0, 1.0)

    order = np.argsort(score)[::-1]
    return np.sort(order[:max_points].astype(np.int64))


def _decode_latents_to_frames(
    latents_path: Path,
    model_name: str,
    config_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    from videox_fun.models import AutoencoderKLWan

    cfg = OmegaConf.load(config_path)
    vae_kwargs = OmegaConf.to_container(cfg["vae_kwargs"])
    if not isinstance(vae_kwargs, dict):
        raise ValueError("`vae_kwargs` must be a dictionary in config.")
    vae_subpath = str(vae_kwargs.get("vae_subpath", "vae"))

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, vae_subpath),
        additional_kwargs=vae_kwargs,
    )
    vae = vae.to(device=device, dtype=dtype).eval()

    latents = torch.load(str(latents_path), map_location="cpu")
    if not isinstance(latents, torch.Tensor):
        raise ValueError(f"Expected Tensor in latent file: {latents_path}")
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"Unexpected latents shape: {tuple(latents.shape)}")

    latents = latents.to(device=device, dtype=dtype)
    with torch.inference_mode():
        decoded = vae.decode(latents).sample

    decoded_video = decoded[0].permute(1, 2, 3, 0)  # [T,H,W,3]
    decoded_video = (decoded_video / 2 + 0.5).clamp(0, 1)
    return (decoded_video * 255.0).round().to(torch.uint8).cpu().numpy()


def _save_mp4(frames_uint8: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=float(fps),
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames_uint8:
            writer.append_data(frame)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay transformed tracks on decoded reference video from VAE latents."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--sample_dir", type=str, default="")
    source_group.add_argument(
        "--metadata_path",
        type=str,
        default="",
        help="Path to json/jsonl/csv metadata containing latent + track paths.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Row index for --metadata_path mode. Negative index is supported.",
    )
    parser.add_argument(
        "--metadata_data_root",
        type=str,
        default="",
        help=(
            "Optional root to prepend to relative metadata paths. "
            "Useful when metadata paths are rooted outside this repo."
        ),
    )
    parser.add_argument(
        "--metadata_track_key",
        type=str,
        default="track_file_path",
        help="Track path key in metadata rows.",
    )
    parser.add_argument(
        "--metadata_latent_key",
        type=str,
        default="latent_file_path",
        help="Latent path key in metadata rows (fallback: file_path).",
    )
    parser.add_argument("--latents_path", type=str, default="")
    parser.add_argument("--track_npz_path", type=str, default="")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.1/wan_civitai.yaml",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP",
    )
    parser.add_argument(
        "--cotracker_root",
        type=str,
        default="/data/project-vilab/jaeseok/co-tracker",
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--decoded_mp4_path", type=str, default="")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--linewidth", type=int, default=3)
    parser.add_argument("--pad_value", type=int, default=120)
    parser.add_argument("--trace_frames", type=int, default=-1)
    parser.add_argument(
        "--max_points",
        type=int,
        default=-1,
        help="Max number of tracks to draw (<=0 means draw all).",
    )
    parser.add_argument(
        "--point_sample_mode",
        type=str,
        default="top_motion",
        choices=["top_motion", "top_visibility", "uniform", "random"],
        help="How to select points when max_points < total points.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    return parser.parse_args()


def main() -> None:
    _ = _add_repo_root_to_syspath()
    args = parse_args()

    cotracker_root = Path(args.cotracker_root).expanduser().resolve()
    cotracker_root_str = str(cotracker_root)
    if cotracker_root_str not in sys.path:
        sys.path.insert(0, cotracker_root_str)
    from cotracker.utils.visualizer import Visualizer

    metadata_row: Optional[Dict] = None
    metadata_idx: Optional[int] = None
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser().resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata path not found: {metadata_path}")
        latents_from_meta, track_from_meta, metadata_row, metadata_idx = _resolve_sample_from_metadata(
            metadata_path=metadata_path,
            sample_index=int(args.sample_index),
            metadata_data_root=str(args.metadata_data_root),
            metadata_track_key=str(args.metadata_track_key),
            metadata_latent_key=str(args.metadata_latent_key),
        )
        sample_dir = latents_from_meta.parent
        latents_path = (
            Path(args.latents_path).expanduser().resolve()
            if args.latents_path
            else latents_from_meta
        )
        track_npz = (
            Path(args.track_npz_path).expanduser().resolve()
            if args.track_npz_path
            else track_from_meta
        )
    else:
        sample_dir = Path(args.sample_dir).expanduser().resolve()
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample dir not found: {sample_dir}")
        latents_path = (
            Path(args.latents_path).expanduser().resolve()
            if args.latents_path
            else (sample_dir / "vae_latents.pt")
        )
        track_npz = _resolve_track_npz(sample_dir=sample_dir, track_npz_path=args.track_npz_path)

    if not latents_path.exists():
        raise FileNotFoundError(f"Latents not found: {latents_path}")
    if not track_npz.exists():
        raise FileNotFoundError(f"Track npz not found: {track_npz}")

    if metadata_row is not None and metadata_idx is not None:
        print(
            "[info] metadata selection "
            f"index={metadata_idx} "
            f"text_head={str(metadata_row.get('text', ''))[:80]!r}"
        )
        print(
            "[info] metadata paths "
            f"track={metadata_row.get(args.metadata_track_key, '')} "
            f"latent={metadata_row.get(args.metadata_latent_key, metadata_row.get('file_path', ''))}"
        )
        print(f"[info] resolved paths track={track_npz} latent={latents_path}")

    tracks, visibility = _load_tracks(track_npz)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to decode Wan latents.")
    device = torch.device("cuda")
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.mixed_precision]

    decoded_frames = _decode_latents_to_frames(
        latents_path=latents_path,
        model_name=args.model_name,
        config_path=args.config_path,
        device=device,
        dtype=dtype,
    )

    t_video = int(decoded_frames.shape[0])
    t_track = int(tracks.shape[0])
    t_vis = int(visibility.shape[0])
    t_common = min(t_video, t_track, t_vis)
    if t_common <= 0:
        raise ValueError(
            f"Invalid time lengths: video={t_video}, tracks={t_track}, visibility={t_vis}"
        )
    if (t_video != t_common) or (t_track != t_common) or (t_vis != t_common):
        print(
            "[warn] temporal length mismatch; truncating to common length: "
            f"{t_common} (video={t_video}, tracks={t_track}, visibility={t_vis})"
        )
    decoded_frames = decoded_frames[:t_common]
    tracks = tracks[:t_common]
    visibility = visibility[:t_common]
    num_points_before = int(tracks.shape[1])
    keep_idx = _select_track_indices(
        tracks=tracks,
        visibility=visibility,
        max_points=int(args.max_points),
        sample_mode=str(args.point_sample_mode),
        seed=int(args.seed),
    )
    tracks = tracks[:, keep_idx]
    visibility = visibility[:, keep_idx]
    print(
        "[info] selected track points: "
        f"{tracks.shape[1]}/{num_points_before} "
        f"(mode={args.point_sample_mode}, max_points={args.max_points})"
    )

    if args.decoded_mp4_path:
        decoded_out = Path(args.decoded_mp4_path).expanduser().resolve()
    else:
        decoded_out = sample_dir / "decoded_from_latents_reference.mp4"
    _save_mp4(decoded_frames, decoded_out, fps=args.fps)
    print(f"[done] saved decoded reference video: {decoded_out}")

    if args.output_path:
        overlay_out = Path(args.output_path).expanduser().resolve()
    else:
        overlay_out = sample_dir / "vis_tracks_on_decoded_reference.mp4"
    overlay_out.parent.mkdir(parents=True, exist_ok=True)

    video_tensor = torch.from_numpy(decoded_frames).permute(0, 3, 1, 2)[None].float().cpu()
    tracks_tensor = torch.from_numpy(tracks).float()[None].cpu()
    vis_tensor = torch.from_numpy(visibility).float()[None].cpu()
    vis = Visualizer(
        save_dir=str(overlay_out.parent),
        pad_value=int(args.pad_value),
        linewidth=int(args.linewidth),
        fps=int(args.fps),
        show_first_frame=0,
        tracks_leave_trace=int(args.trace_frames),
    )
    vis.visualize(
        video=video_tensor,
        tracks=tracks_tensor,
        visibility=vis_tensor,
        filename=overlay_out.stem,
        query_frame=0,
        save_video=True,
    )
    print(f"[done] saved overlay video: {overlay_out}")


if __name__ == "__main__":
    main()

