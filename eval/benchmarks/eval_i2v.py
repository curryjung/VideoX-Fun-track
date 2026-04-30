"""
Image-to-Video (I2V) Benchmark — PSNR / SSIM / LPIPS / CLIP Score

Evaluates visual fidelity and text alignment when generating videos from
a conditioning image (no track conditioning).

Usage:
  cd /data/project-vilab/jaeseok/VideoX-Fun
  python eval/benchmarks/eval_i2v.py \
    --model_name models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP \
    --config_path config/wan2.1/wan_civitai.yaml \
    --dataset davis \
    --output_dir eval/results/i2v
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
from metrics.clip_score import compute_clip_score, load_clip
from utils.track_extraction import load_video_frames


def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames], axis=0
    )


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


def parse_args():
    p = argparse.ArgumentParser(description="I2V Evaluation")
    p.add_argument("--model_name", required=True)
    p.add_argument("--config_path", default="config/wan2.1/wan_civitai.yaml")
    p.add_argument("--transformer_checkpoint_path", default=None)
    p.add_argument("--dataset", choices=["davis", "sora_subset", "both"], default="davis")
    p.add_argument("--davis_root",
                   default=str(_EVAL_ROOT / "data" / "davis"))
    p.add_argument("--sora_root",
                   default=str(_EVAL_ROOT / "data" / "sora_subset"))
    p.add_argument("--output_dir",
                   default=str(_EVAL_ROOT / "results" / "i2v"))
    p.add_argument("--prompts_json", default=None)
    p.add_argument("--sample_height", type=int, default=480)
    p.add_argument("--sample_width", type=int, default=832)
    p.add_argument("--video_length", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--max_videos", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_roots = {"davis": args.davis_root, "sora_subset": args.sora_root}
    selected = ["davis", "sora_subset"] if args.dataset == "both" else [args.dataset]

    from utils.video_generation import VideoGenerator
    generator = VideoGenerator(
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
        normalize_track=False,  # no track conditioning in I2V
    )
    lpips_fn = load_lpips(device)
    clip_model, clip_processor = load_clip(device=device)

    prompts = {}
    if args.prompts_json and Path(args.prompts_json).is_file():
        with open(args.prompts_json) as f:
            prompts = json.load(f)

    for dataset_name in selected:
        data_root = dataset_roots[dataset_name]
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

        videos = discover_videos(data_root)
        if args.max_videos:
            videos = videos[: args.max_videos]

        out_dir = Path(args.output_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_list = []

        for vid in videos:
            name = vid.stem if vid.is_file() else vid.name
            prompt = prompts.get(name, "")
            print(f"\n── {name} ──")

            gt_frames = resize_frames(
                load_video_frames(str(vid)), args.sample_height, args.sample_width
            )
            first_frame_path = str(out_dir / f"{name}_first.png")
            cv2.imwrite(first_frame_path, cv2.cvtColor(gt_frames[0], cv2.COLOR_RGB2BGR))

            gen_path = str(out_dir / f"{name}_gen.mp4")
            generator.generate(
                first_frame_path=first_frame_path,
                track_npz_path="",  # no tracks for I2V
                prompt=prompt,
                output_path=gen_path,
                seed=args.seed,
                fps=args.fps,
            )

            cap = cv2.VideoCapture(gen_path)
            gen_raw = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                gen_raw.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            gen_frames = resize_frames(np.stack(gen_raw, axis=0), args.sample_height, args.sample_width)

            T = min(gt_frames.shape[0], gen_frames.shape[0])
            vf = compute_visual_metrics(gt_frames[:T], gen_frames[:T], lpips_fn=lpips_fn, device=device)
            clip_s = compute_clip_score(gen_frames, prompt, clip_model, clip_processor, device)
            print(f"  PSNR={vf['psnr']:.3f}  SSIM={vf['ssim']:.4f}  LPIPS={vf['lpips']:.4f}  CLIP={clip_s:.4f}")

            m = {"name": name, **vf, "clip_score": clip_s}
            metrics_list.append(m)
            with open(out_dir / f"{name}_metrics.json", "w") as f:
                json.dump(m, f, indent=2)

        keys = ["psnr", "ssim", "lpips", "clip_score"]
        agg = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if not np.isnan(m.get(k, float("nan")))]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg["num_videos"] = len(metrics_list)

        print(f"\n[{dataset_name}] PSNR={agg['psnr']:.3f}  SSIM={agg['ssim']:.4f}  "
              f"LPIPS={agg['lpips']:.4f}  CLIP={agg['clip_score']:.4f}")
        with open(out_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

    print("\n[done]")


if __name__ == "__main__":
    main()
