from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.cotracker_utils import (  # noqa: E402
    build_uniform_grid_queries,
    discover_davis_videos,
    extract_tracks_at_queries,
    load_cotracker,
    load_video_frames,
    prepare_davis_frames,
)
from evaluation.metrics import build_lpips_model, compute_metric  # noqa: E402
from evaluation.track_io import (  # noqa: E402
    TrackSampleConfig,
    load_tracks_npz,
    sample_track_subset,
    save_tracks_npz,
)
from evaluation.visualization import save_track_overlay_video  # noqa: E402


DEFAULT_DAVIS_ROOT = "/data/project-vilab/jaeseok/davis/DAVIS/JPEGImages/480p"
DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, blurry, static frame"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_prompts(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("prompts_json must be a JSON object mapping video name to prompt.")
    return {str(k): str(v) for k, v in data.items()}


def load_generated_frames(path: str | Path, height: int, width: int) -> np.ndarray:
    return load_video_frames(path, target_hw=(height, width))


def video_tensor_to_rgb_frames(video: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        arr = video.detach().float().cpu().numpy()
    else:
        arr = np.asarray(video)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected one generated batch, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected generated video with 4 or 5 dims, got {arr.shape}")
    if arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 3, 0))
    elif arr.shape[1] in {1, 3}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[-1] not in {1, 3}:
        raise ValueError(f"Could not infer generated video channel axis from shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmin(arr)) < -0.1:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.rint(arr * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr[..., :3])


def load_generated_frames_npy(path: str | Path, height: int, width: int) -> np.ndarray:
    frames = np.load(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected generated_frames.npy with shape (T, H, W, 3), got {frames.shape}")
    if frames.shape[1] != int(height) or frames.shape[2] != int(width):
        raise ValueError(
            "generated_frames.npy resolution mismatch: "
            f"got {frames.shape[2]}x{frames.shape[1]}, expected {width}x{height}"
        )
    return np.ascontiguousarray(frames.astype(np.uint8, copy=False))


def save_rgb_video_visualization(path: Path, frames_rgb: np.ndarray, fps: int) -> str:
    frames_rgb = np.asarray(frames_rgb, dtype=np.uint8)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape (T, H, W, 3), got {frames_rgb.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "0",
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            subprocess.run(cmd, input=np.ascontiguousarray(frames_rgb).tobytes(), check=True)
            return "libx264_yuv420p_crf0"
        except subprocess.CalledProcessError:
            print("  warning: ffmpeg generated video encode failed; falling back to OpenCV mp4v")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return "mp4v_fallback"


def write_first_frame(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write first frame: {path}")


def finite_mean(values: list[float]) -> float:
    kept = [v for v in values if not math.isnan(float(v))]
    if not kept:
        return float("nan")
    return float(np.mean(kept))


def aggregate(metrics_list: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_videos": len(metrics_list)}
    for name in metric_names:
        out[name] = finite_mean([float(item.get(name, float("nan"))) for item in metrics_list])
    return out


def safe_path_tag(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def cotracker_cache_tag(cotracker_checkpoint: str | None) -> str:
    if not cotracker_checkpoint:
        return "cotracker_default"
    normalized = str(Path(cotracker_checkpoint).expanduser().resolve(strict=False))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"cotracker_{digest}"


def get_gt_track_cache_path(name: str, args: argparse.Namespace) -> Path | None:
    if not args.gt_track_cache_dir:
        return None
    filename = (
        f"{safe_path_tag(name)}"
        f"_h{args.sample_height}_w{args.sample_width}"
        f"_vl{args.video_length}_grid{args.grid_size}.npz"
    )
    return Path(args.gt_track_cache_dir) / cotracker_cache_tag(args.cotracker_checkpoint) / filename


def load_valid_gt_track_cache(
    path: Path,
    *,
    expected_query_points: np.ndarray,
    expected_frame_indices: np.ndarray,
    expected_num_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None

    try:
        tracks, visibility = load_tracks_npz(path)
        with np.load(path) as data:
            if "query_points" not in data or "frame_indices" not in data:
                print(f"  ignoring GT track cache without metadata: {path}")
                return None
            cached_query_points = np.asarray(data["query_points"], dtype=np.float32)
            cached_frame_indices = np.asarray(data["frame_indices"])
    except Exception as exc:
        print(f"  ignoring unreadable GT track cache: {path} ({exc})")
        return None

    if tracks.shape[0] != expected_num_frames or visibility.shape[0] != expected_num_frames:
        print(f"  ignoring GT track cache with mismatched frame count: {path}")
        return None
    if tracks.shape[1] != expected_query_points.shape[0]:
        print(f"  ignoring GT track cache with mismatched point count: {path}")
        return None
    if cached_query_points.shape != expected_query_points.shape or not np.allclose(
        cached_query_points, expected_query_points
    ):
        print(f"  ignoring GT track cache with mismatched query grid: {path}")
        return None
    if cached_frame_indices.shape != expected_frame_indices.shape or not np.array_equal(
        cached_frame_indices, expected_frame_indices
    ):
        print(f"  ignoring GT track cache with mismatched frame indices: {path}")
        return None
    return tracks, visibility


def evaluate_video(
    *,
    video_path: Path,
    prompt: str,
    generator,
    cotracker,
    lpips_model,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = video_path.stem if video_path.is_file() else video_path.name
    print(f"\n[evaluation] {name}")
    sample_dir = output_dir / name
    sample_dir.mkdir(parents=True, exist_ok=True)

    full_frames = load_video_frames(
        video_path,
        target_hw=(args.sample_height, args.sample_width),
    )
    selected_frames, frame_indices = prepare_davis_frames(full_frames, args.video_length)
    print(
        f"  frames: full={full_frames.shape[0]}, selected={selected_frames.shape[0]}, "
        f"resolution={args.sample_width}x{args.sample_height}"
    )

    query_points = build_uniform_grid_queries(
        selected_frames.shape[1],
        selected_frames.shape[2],
        grid_size=args.grid_size,
    )
    gt_tracks_path = sample_dir / "gt_tracks_full.npz"
    gt_tracks_cache_path = get_gt_track_cache_path(name, args)
    gt_tracks_source = "extracted"
    gt_tracks = gt_visibility = None

    cache_candidates: list[Path] = []
    if not args.overwrite_gt_track_cache:
        if gt_tracks_cache_path is not None:
            cache_candidates.append(gt_tracks_cache_path)
        cache_candidates.append(gt_tracks_path)

    for cache_path in cache_candidates:
        cached = load_valid_gt_track_cache(
            cache_path,
            expected_query_points=query_points,
            expected_frame_indices=frame_indices,
            expected_num_frames=selected_frames.shape[0],
        )
        if cached is not None:
            gt_tracks, gt_visibility = cached
            gt_tracks_source = str(cache_path)
            print(f"  loaded GT tracks: {cache_path}")
            break

    if gt_tracks is None or gt_visibility is None:
        print(f"  extracting GT tracks: grid={args.grid_size} points={len(query_points)}")
        gt_tracks, gt_visibility = extract_tracks_at_queries(
            selected_frames,
            query_points,
            model=cotracker,
        )
    else:
        print(f"  GT tracks cached: grid={args.grid_size} points={len(query_points)}")

    if gt_tracks_source != str(gt_tracks_path):
        save_tracks_npz(
            gt_tracks_path,
            gt_tracks,
            gt_visibility,
            query_points=query_points,
            frame_indices=frame_indices,
        )
    if gt_tracks_cache_path is not None and gt_tracks_source != str(gt_tracks_cache_path):
        save_tracks_npz(
            gt_tracks_cache_path,
            gt_tracks,
            gt_visibility,
            query_points=query_points,
            frame_indices=frame_indices,
        )

    effective_track_seed = args.seed if args.track_point_sample_seed is None else args.track_point_sample_seed
    sample_config = TrackSampleConfig(
        max_points=args.track_max_points,
        sample_mode=args.track_point_sample_mode,
        sort_selected_indices=args.track_sort_selected_indices,
        seed=effective_track_seed,
        point_id_mode=args.track_point_id_mode,
    )
    condition_tracks, condition_visibility, keep_idx, point_ids = sample_track_subset(
        gt_tracks,
        gt_visibility,
        sample_config,
    )
    condition_queries = query_points[keep_idx]
    save_tracks_npz(
        sample_dir / "condition_tracks.npz",
        condition_tracks,
        condition_visibility,
        query_points=condition_queries,
        selected_indices=keep_idx,
        point_ids=point_ids,
        frame_indices=frame_indices,
    )
    print(f"  condition tracks: {condition_tracks.shape[1]} points")
    condition_overlay_path = sample_dir / "condition_tracks_overlay.mp4"
    save_track_overlay_video(
        selected_frames,
        condition_tracks,
        condition_visibility,
        condition_overlay_path,
        fps=args.fps,
        trace_frames=args.track_overlay_trace_frames,
        scale=args.track_overlay_scale,
        crf=args.track_overlay_crf,
    )

    first_frame_path = sample_dir / "first_frame.png"
    write_first_frame(first_frame_path, selected_frames[0])
    generated_path = sample_dir / "generated.mp4"
    generated_frames_path = sample_dir / "generated_frames.npy"
    reuse_generated = args.reuse_generated_frames.lower()
    if reuse_generated in {"auto", "always"} and generated_frames_path.is_file():
        print(f"  loading generated frames: {generated_frames_path}")
        generated_frames = load_generated_frames_npy(
            generated_frames_path,
            height=args.sample_height,
            width=args.sample_width,
        )
        generated_length = int(generated_frames.shape[0])
    else:
        if reuse_generated == "always":
            raise FileNotFoundError(f"generated_frames.npy not found: {generated_frames_path}")
        print("  generating video")
        generated_sample, generated_length = generator.generate(
            first_frame_path=str(first_frame_path),
            tracks=condition_tracks,
            visibility=condition_visibility,
            output_path=None,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            fps=args.fps,
            video_length=selected_frames.shape[0],
            point_ids=point_ids,
            save_video=False,
        )
        generated_frames = video_tensor_to_rgb_frames(generated_sample)
        np.save(generated_frames_path, generated_frames)
        print(f"  generated frames saved: {generated_frames_path}")

    generated_video_codec = save_rgb_video_visualization(generated_path, generated_frames, fps=args.fps)
    print(f"  generated frames: {generated_frames.shape[0]} requested={generated_length}")
    generated_condition_overlay_path = sample_dir / "generated_condition_overlay.mp4"
    save_track_overlay_video(
        generated_frames,
        condition_tracks,
        condition_visibility,
        generated_condition_overlay_path,
        fps=args.fps,
        trace_frames=args.track_overlay_trace_frames,
        scale=args.track_overlay_scale,
        crf=args.track_overlay_crf,
    )

    print("  re-extracting generated tracks")
    generated_tracks, generated_visibility = extract_tracks_at_queries(
        generated_frames,
        condition_queries,
        model=cotracker,
    )
    save_tracks_npz(
        sample_dir / "generated_tracks.npz",
        generated_tracks,
        generated_visibility,
        query_points=condition_queries,
        selected_indices=keep_idx,
        point_ids=point_ids,
    )

    frame_count = min(selected_frames.shape[0], generated_frames.shape[0])
    gt_eval = selected_frames[:frame_count]
    generated_eval = generated_frames[:frame_count]
    uses_lpips = "lpips" in {metric.lower() for metric in args.metrics}
    metric_values: dict[str, float] = {}
    for metric_name in args.metrics:
        metric_values[metric_name] = compute_metric(
            metric_name,
            gt_frames=gt_eval,
            generated_frames=generated_eval,
            input_tracks=condition_tracks,
            input_visibility=condition_visibility,
            generated_tracks=generated_tracks,
            lpips_model=lpips_model,
            lpips_net=args.lpips_net,
            lpips_device=args.lpips_device,
            lpips_batch_size=args.lpips_batch_size,
        )
        print(f"  {metric_name}={metric_values[metric_name]:.6g}")

    result: dict[str, Any] = {
        "name": name,
        **metric_values,
        "video_path": str(video_path),
        "generated_path": str(generated_path),
        "generated_frames_path": str(generated_frames_path),
        "reuse_generated_frames": reuse_generated,
        "generated_video_codec": generated_video_codec,
        "condition_overlay_path": str(condition_overlay_path),
        "generated_condition_overlay_path": str(generated_condition_overlay_path),
        "T_full": int(full_frames.shape[0]),
        "T_selected": int(selected_frames.shape[0]),
        "T_generated": int(generated_frames.shape[0]),
        "T_eval": int(frame_count),
        "grid_size": int(args.grid_size),
        "num_grid_points": int(query_points.shape[0]),
        "num_condition_points": int(condition_tracks.shape[1]),
        "gt_tracks_path": str(gt_tracks_path),
        "gt_tracks_cache_path": str(gt_tracks_cache_path or ""),
        "gt_tracks_source": gt_tracks_source,
        "track_max_points": int(args.track_max_points),
        "track_point_sample_mode": args.track_point_sample_mode,
        "track_point_sample_seed": effective_track_seed,
        "track_sort_selected_indices": bool(args.track_sort_selected_indices),
        "track_point_id_mode": args.track_point_id_mode,
        "track_latent_scale": float(args.track_latent_scale),
        "track_latent_first_frame_scale": args.track_latent_first_frame_scale,
        "track_latent_rest_frame_scale": args.track_latent_rest_frame_scale,
        "track_overlay_scale": float(args.track_overlay_scale),
        "track_overlay_crf": args.track_overlay_crf,
        "lpips_net": args.lpips_net if uses_lpips else "",
        "lpips_batch_size": int(args.lpips_batch_size),
        "lpips_device": args.lpips_device if uses_lpips else "",
        "negative_text_feature_path": args.negative_text_feature_path or "",
        "prompt": prompt,
        "checkpoint": args.transformer_checkpoint_path or "",
    }
    with (sample_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAVIS track-conditioned video evaluation")
    parser.add_argument("--davis_root", default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--output_dir", default="evaluation/results/davis_track")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--config_path", default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--transformer_checkpoint_path", default=None)
    parser.add_argument("--prompts_json", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--negative_text_feature_path",
        default=None,
        help="Optional precomputed negative text feature npz with `prompt_embeds`.",
    )
    parser.add_argument("--metrics", nargs="+", default=["epe", "psnr"], choices=["epe", "psnr", "ssim", "lpips"])
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips_batch_size", type=int, default=16)
    parser.add_argument(
        "--lpips_device",
        default=None,
        help="Device for LPIPS model. Defaults to the evaluation device.",
    )
    parser.add_argument("--cotracker_checkpoint", default=None)
    parser.add_argument(
        "--gt_track_cache_dir",
        default=None,
        help="Optional shared cache dir for CoTracker GT tracks. Existing valid npz files skip extraction.",
    )
    parser.add_argument(
        "--overwrite_gt_track_cache",
        type=str2bool,
        default=False,
        help="Re-extract GT tracks even when a valid cached npz exists.",
    )
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--sample_height", type=int, default=480)
    parser.add_argument("--sample_width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument(
        "--reuse_generated_frames",
        default="never",
        choices=["never", "auto", "always"],
        help="Reuse sample_dir/generated_frames.npy instead of running generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--mixed_precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance_mode", default="motion_only", choices=["cfg", "joint_tm", "text_only", "motion_only", "unified"])
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--text_guidance_weight", type=float, default=0.0)
    parser.add_argument("--motion_guidance_weight", type=float, default=3.5)
    parser.add_argument("--sampler_name", default="Flow", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--normalize_track", type=str2bool, default=True)
    parser.add_argument("--track_normalize_height", type=int, default=480)
    parser.add_argument("--track_normalize_width", type=int, default=832)
    parser.add_argument("--track_latent_scale", type=float, default=1.0)
    parser.add_argument("--track_latent_first_frame_scale", type=float, default=None)
    parser.add_argument("--track_latent_rest_frame_scale", type=float, default=None)
    parser.add_argument("--track_head_hidden_dim", type=int, default=None)
    parser.add_argument("--track_condition_mode", default="track_head", choices=["track_head", "wan_move"])
    parser.add_argument("--wan_move_temporal_stride", type=int, default=0)
    parser.add_argument("--track_max_points", type=int, default=-1)
    parser.add_argument("--track_point_sample_mode", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--track_sort_selected_indices", type=str2bool, default=True)
    parser.add_argument("--track_point_sample_seed", type=int, default=None)
    parser.add_argument("--track_point_id_mode", default="original", choices=["original", "local"])
    parser.add_argument(
        "--track_overlay_trace_frames",
        type=int,
        default=8,
        help="Number of recent track positions to draw in overlay videos (-1 = full history).",
    )
    parser.add_argument(
        "--track_overlay_scale",
        type=float,
        default=0.5,
        help="Spatial scale for saved overlay videos. Use 1.0 for full resolution.",
    )
    parser.add_argument(
        "--track_overlay_crf",
        type=int,
        default=32,
        help="ffmpeg H.264 CRF for overlay videos. Lower is larger/better; set -1 to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.track_overlay_crf is not None and args.track_overlay_crf < 0:
        args.track_overlay_crf = None
    if args.lpips_batch_size <= 0:
        raise ValueError("--lpips_batch_size must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_davis_videos(args.davis_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No DAVIS videos found under: {args.davis_root}")

    prompts = load_prompts(args.prompts_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.lpips_device = args.lpips_device or str(device)
    print(f"[evaluation] device={device} videos={len(videos)} output_dir={output_dir}")

    cotracker = load_cotracker(args.cotracker_checkpoint, device=device)
    lpips_model = None
    if "lpips" in {metric.lower() for metric in args.metrics}:
        print(f"[evaluation] loading LPIPS: net={args.lpips_net} device={args.lpips_device}")
        lpips_model = build_lpips_model(net=args.lpips_net, device=args.lpips_device)

    from evaluation.generator import TrackConditionedVideoGenerator

    generator = TrackConditionedVideoGenerator(
        model_name=args.model_name,
        config_path=args.config_path,
        transformer_checkpoint_path=args.transformer_checkpoint_path,
        mixed_precision=args.mixed_precision,
        device=str(device),
        sample_height=args.sample_height,
        sample_width=args.sample_width,
        video_length=args.video_length,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_mode=args.guidance_mode,
        text_guidance_weight=args.text_guidance_weight,
        motion_guidance_weight=args.motion_guidance_weight,
        negative_text_feature_path=args.negative_text_feature_path,
        sampler_name=args.sampler_name,
        shift=args.shift,
        normalize_track=args.normalize_track,
        track_normalize_height=args.track_normalize_height,
        track_normalize_width=args.track_normalize_width,
        track_latent_scale=args.track_latent_scale,
        track_latent_first_frame_scale=args.track_latent_first_frame_scale,
        track_latent_rest_frame_scale=args.track_latent_rest_frame_scale,
        track_head_hidden_dim=args.track_head_hidden_dim,
        track_condition_mode=args.track_condition_mode,
        wan_move_temporal_stride=args.wan_move_temporal_stride,
    )

    results: list[dict[str, Any]] = []
    for video_path in videos:
        name = video_path.stem if video_path.is_file() else video_path.name
        prompt = prompts.get(name, args.prompt)
        result = evaluate_video(
            video_path=video_path,
            prompt=prompt,
            generator=generator,
            cotracker=cotracker,
            lpips_model=lpips_model,
            output_dir=output_dir,
            args=args,
        )
        results.append(result)

    aggregate_result = aggregate(results, args.metrics)
    with (output_dir / "per_video.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_result, f, indent=2)
    write_csv(output_dir / "per_video.csv", results)

    print("\n[evaluation] aggregate")
    for key, value in aggregate_result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
