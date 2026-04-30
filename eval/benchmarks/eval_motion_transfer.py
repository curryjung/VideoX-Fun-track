"""
Motion Transfer Benchmark — PSNR / SSIM / LPIPS / EPE

Evaluation protocol (Section 4.2 + Appendix E of the Self Forcing paper):

  Frame handling
  ─────────────
  • DAVIS  : if GT frames > model_video_length (81), retain first and last frames
             and uniformly subsample intermediate frames to fit.
             Pixel metrics compare the same subsampled GT frames vs. generated frames.
  • Sora   : cap at 81 frames (first 81 only).

  Track protocol (Appendix E)
  ───────────────────────────
  • 50×50 uniform grid on the first frame (= 2500 query points).
  • CoTracker3 (offline) extracts tracks from the (subsampled) GT video → used as
    model input conditioning.
  • After generation, CoTracker3 re-extracts tracks from the generated video at the
    same query points.
  • EPE = mean L2 distance between visible GT tracks and re-extracted tracks.
  • All coordinates live in 832×480 pixel space (resize before tracking).

  CoTracker3 is loaded once per run and reused across all videos.

Dataset layout:
  eval/data/davis/<video_name>/       (frame images dir or .mp4)
  eval/data/sora_subset/<video_name>/

Usage:
  cd /data/project-vilab/jaeseok/VideoX-Fun
  python eval/benchmarks/eval_motion_transfer.py \
    --model_name models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP \
    --config_path config/wan2.1/wan_civitai.yaml \
    --transformer_checkpoint_path checkpoints/<run>/checkpoint-<step> \
    --dataset both \
    --prompts_json eval/data/prompts.json \
    [--cotracker_checkpoint /path/to/scaled_offline.pth]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _BENCHMARKS_DIR.parent
_PROJECT_ROOT = _EVAL_ROOT.parent
for _p in [str(_EVAL_ROOT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics.visual_fidelity import compute_all as compute_visual_metrics, load_lpips
from metrics.epe import compute_epe
from utils.track_extraction import (
    load_cotracker,
    extract_tracks_at_queries,
    build_uniform_grid_queries,
    load_video_frames,
    save_tracks_npz,
)
from utils.video_generation import VideoGenerator


# ── Frame subsampling (Appendix E) ───────────────────────────────────────────

def subsample_frames(frames: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample a video to target_len frames using the paper's protocol:
      retain first and last frames, uniformly subsample intermediate frames.

    Args:
        frames: (T, H, W, 3) uint8 array.
        target_len: desired number of frames.

    Returns:
        (subsampled_frames, indices) where indices are the selected frame indices
        from the original video (used to select the matching GT track frames).
    """
    T = frames.shape[0]
    if T <= target_len:
        return frames, np.arange(T)

    # First + last retained; intermediate uniformly sampled
    middle_indices = np.linspace(1, T - 2, target_len - 2, dtype=float)
    middle_indices = np.round(middle_indices).astype(int)
    indices = np.concatenate([[0], middle_indices, [T - 1]])
    indices = np.unique(indices)  # remove duplicates if T is very small

    # If unique collapsed below target (edge case), pad with nearest
    if len(indices) < target_len:
        extra = np.linspace(0, T - 1, target_len, dtype=float)
        indices = np.unique(np.round(extra).astype(int))

    return frames[indices], indices


def prepare_gt_frames(
    frames_full: np.ndarray,
    dataset: str,
    model_video_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dataset-specific frame selection before tracking and generation.

    Returns:
        (selected_frames, frame_indices) into frames_full.
    """
    T = frames_full.shape[0]

    if dataset == "sora_subset":
        # Cap at model_video_length (paper: 81 frames)
        end = min(T, model_video_length)
        return frames_full[:end], np.arange(end)

    # DAVIS: subsample when longer than model_video_length
    if T > model_video_length:
        return subsample_frames(frames_full, model_video_length)

    return frames_full, np.arange(T)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def discover_videos(data_root: str) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(f.suffix.lower() in img_exts for f in p.iterdir()):
            videos.append(p)
        elif p.suffix.lower() in video_exts:
            videos.append(p)
    return videos


def load_prompts(prompts_json: str | None, names: list[str]) -> dict[str, str]:
    if prompts_json and Path(prompts_json).is_file():
        with open(prompts_json) as f:
            return json.load(f)
    print("[eval] No prompts JSON — using empty prompt for all videos.")
    return {n: "" for n in names}


def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames], axis=0
    )


def load_mp4_frames(path: str, h: int, w: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read video: {path}")
    return resize_frames(np.stack(frames, axis=0), h, w)


# ── Per-video evaluation ──────────────────────────────────────────────────────

def evaluate_video(
    video_path: Path,
    dataset: str,
    prompt: str,
    generator: VideoGenerator,
    cotracker,
    output_dir: Path,
    args: argparse.Namespace,
    lpips_fn,
    device: torch.device,
) -> dict:
    name = video_path.stem if video_path.is_file() else video_path.name
    print(f"\n── {name} ──")
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    
    gt_full = resize_frames(
        load_video_frames(str(video_path)),
        args.sample_height, args.sample_width,
    )
    T_full = gt_full.shape[0]

    # 2. Apply dataset-specific frame selection (DAVIS subsample / Sora cap)
    gt_frames, _ = prepare_gt_frames(gt_full, dataset, args.model_video_length)
    T_gt = gt_frames.shape[0]
    print(f"  GT: {T_full} frames → using {T_gt} (dataset={dataset})")

    # 3. Build 50×50 uniform query grid and extract GT tracks from selected frames
    H, W = gt_frames.shape[1], gt_frames.shape[2]
    query_pts = build_uniform_grid_queries(H, W, grid_size=args.grid_size)
    print(f"  Extracting GT tracks ({args.grid_size}²={len(query_pts)} points)...")
    gt_tracks, gt_vis = extract_tracks_at_queries(gt_frames, query_pts, model=cotracker)
    # gt_tracks: (T_gt, N, 2) in 832×480 pixel space

    gt_track_npz = str(out / "gt_tracks.npz")
    save_tracks_npz(gt_track_npz, gt_tracks, gt_vis)

    # 4. Save first frame for the generator
    first_frame_path = str(out / "first_frame.png")
    cv2.imwrite(first_frame_path, cv2.cvtColor(gt_frames[0], cv2.COLOR_RGB2BGR))

    # 5. Generate video conditioned on GT tracks.
    #    Pass T_gt so the generated length matches the GT clip exactly when the
    #    video is shorter than the model's default temporal length (e.g. short
    #    DAVIS clips).  For videos at or above model_video_length, T_gt already
    #    equals model_video_length after subsampling/capping above.
    gen_video_path = str(out / "generated.mp4")
    print("  Generating video...")
    generator.generate(
        first_frame_path=first_frame_path,
        track_npz_path=gt_track_npz,
        prompt=prompt,
        output_path=gen_video_path,
        seed=args.seed,
        fps=args.fps,
        video_length=T_gt,
    )

    # 6. Load generated frames (already at 832×480 from pipeline; resize to be safe)
    gen_frames = load_mp4_frames(gen_video_path, args.sample_height, args.sample_width)
    T_gen = gen_frames.shape[0]
    print(f"  Generated: {T_gen} frames")

    # 7. Align frame counts for pixel metrics
    #    Compare GT subsampled frames vs generated frames (both already at 832×480)
    T_eval = min(T_gt, T_gen)
    gt_eval = gt_frames[:T_eval]
    gen_eval = gen_frames[:T_eval]

    vf = compute_visual_metrics(gt_eval, gen_eval, lpips_fn=lpips_fn, device=device)
    print(f"  PSNR={vf['psnr']:.3f}  SSIM={vf['ssim']:.4f}  LPIPS={vf['lpips']:.4f}")

    # 8. Re-extract tracks from generated video at the same query points
    print("  Re-extracting tracks from generated video...")
    gen_tracks, _ = extract_tracks_at_queries(gen_frames, query_pts, model=cotracker)
    # gen_tracks: (T_gen, N, 2) in 832×480 pixel space

    # 9. EPE: visible GT tracks vs re-extracted tracks (Appendix E)
    #    Both track arrays are in 832×480 pixel space → EPE in pixels
    epe = compute_epe(gt_tracks, gt_vis, gen_tracks)
    print(f"  EPE={epe:.4f} px")

    metrics = {
        "name": name,
        **vf,
        "epe": epe,
        "T_full": T_full,
        "T_gt": T_gt,
        "T_gen": T_gen,
        "T_eval": T_eval,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(all_metrics: list[dict]) -> dict:
    keys = ["psnr", "ssim", "lpips", "epe"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    agg["num_videos"] = len(all_metrics)
    return agg


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Motion Transfer Evaluation")
    p.add_argument("--model_name", required=True,
                   help="Path to Wan2.1-Fun I2V model directory.")
    p.add_argument("--config_path", default="config/wan2.1/wan_civitai.yaml")
    p.add_argument("--transformer_checkpoint_path", default=None)
    p.add_argument("--dataset", choices=["davis", "sora_subset", "both"], default="both")
    p.add_argument("--davis_root",
                   default=str(_EVAL_ROOT / "data" / "davis"))
    p.add_argument("--sora_root",
                   default=str(_EVAL_ROOT / "data" / "sora_subset"))
    p.add_argument("--output_dir",
                   default=str(_EVAL_ROOT / "results" / "motion_transfer"))
    p.add_argument("--prompts_json", default=None,
                   help='JSON mapping {"video_name": "prompt", ...}')
    p.add_argument("--cotracker_checkpoint", default=None,
                   help="Path to CoTracker3 .pth checkpoint (auto-detected if omitted).")
    p.add_argument("--grid_size", type=int, default=50,
                   help="Uniform grid side length — paper uses 50 (→ 2500 points).")
    p.add_argument("--sample_height", type=int, default=480,
                   help="Evaluation height — paper: 480.")
    p.add_argument("--sample_width", type=int, default=832,
                   help="Evaluation width — paper: 832.")
    p.add_argument("--model_video_length", type=int, default=81,
                   help="Model temporal length. DAVIS frames are subsampled to this "
                        "when the video is longer; Sora videos are capped at this.")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--guidance_mode", default="cfg")
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--normalize_track", action="store_true", default=True)
    p.add_argument("--max_videos", type=int, default=None,
                   help="Cap number of videos per dataset (for quick debugging).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_roots = {
        "davis": args.davis_root,
        "sora_subset": args.sora_root,
    }
    selected = ["davis", "sora_subset"] if args.dataset == "both" else [args.dataset]

    # Load VideoGenerator and CoTracker3 once — reused across all videos
    generator = VideoGenerator(
        model_name=args.model_name,
        config_path=args.config_path,
        transformer_checkpoint_path=args.transformer_checkpoint_path,
        mixed_precision=args.mixed_precision,
        device=str(device),
        sample_height=args.sample_height,
        sample_width=args.sample_width,
        video_length=args.model_video_length,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_mode=args.guidance_mode,
        normalize_track=args.normalize_track,
    )
    print("[eval] Loading CoTracker3...")
    cotracker = load_cotracker(args.cotracker_checkpoint, device)
    lpips_fn = load_lpips(device)

    all_results = {}

    for dataset_name in selected:
        data_root = dataset_roots[dataset_name]
        print(f"\n{'='*60}\nDataset: {dataset_name}  ({data_root})\n{'='*60}")

        videos = discover_videos(data_root)
        if not videos:
            print("  No videos found — skipping.")
            continue
        if args.max_videos:
            videos = videos[: args.max_videos]

        names = [v.stem if v.is_file() else v.name for v in videos]
        prompts = load_prompts(args.prompts_json, names)
        out_dir = Path(args.output_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_list = []
        for vid in videos:
            vid_name = vid.stem if vid.is_file() else vid.name
            try:
                m = evaluate_video(
                    video_path=vid,
                    dataset=dataset_name,
                    prompt=prompts.get(vid_name, ""),
                    generator=generator,
                    cotracker=cotracker,
                    output_dir=out_dir,
                    args=args,
                    lpips_fn=lpips_fn,
                    device=device,
                )
                metrics_list.append(m)
            except Exception as exc:
                import traceback
                print(f"  [ERROR] {vid_name}: {exc}")
                traceback.print_exc()

        agg = aggregate(metrics_list)
        all_results[dataset_name] = {"per_video": metrics_list, "aggregate": agg}

        print(f"\n[{dataset_name}] {agg['num_videos']} videos:")
        print(f"  PSNR  = {agg['psnr']:.3f}")
        print(f"  SSIM  = {agg['ssim']:.4f}")
        print(f"  LPIPS = {agg['lpips']:.4f}")
        print(f"  EPE   = {agg['epe']:.4f} px")

        with open(out_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)
        with open(out_dir / "per_video.json", "w") as f:
            json.dump(metrics_list, f, indent=2)

    summary = Path(args.output_dir) / "summary.json"
    with open(summary, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] Summary → {summary}")


if __name__ == "__main__":
    main()
