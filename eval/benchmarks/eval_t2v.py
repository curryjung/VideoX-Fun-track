"""
Text-to-Video (T2V) Benchmark — FVD + CLIP Score

Evaluates the quality and text alignment of generated videos given only text prompts
(no conditioning image or tracks).

Metrics:
  - FVD  : Fréchet Video Distance between GT and generated distributions.
  - CLIP : Mean cosine similarity between text prompt and generated frames.

Usage:
  cd /data/project-vilab/jaeseok/VideoX-Fun
  python eval/benchmarks/eval_t2v.py \
    --model_name models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP \
    --config_path config/wan2.1/wan_civitai.yaml \
    --prompts_json eval/data/t2v_prompts.json \
    --output_dir eval/results/t2v
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _BENCHMARKS_DIR.parent
_PROJECT_ROOT = _EVAL_ROOT.parent
for _p in [str(_EVAL_ROOT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics.fvd import compute_fvd, load_i3d
from metrics.clip_score import compute_clip_score, load_clip


def parse_args():
    p = argparse.ArgumentParser(description="T2V Evaluation")
    p.add_argument("--model_name", required=True)
    p.add_argument("--config_path", default="config/wan2.1/wan_civitai.yaml")
    p.add_argument("--transformer_checkpoint_path", default=None)
    p.add_argument("--prompts_json", required=True,
                   help='JSON: [{"name": "...", "prompt": "..."}, ...]')
    p.add_argument("--gt_root", default=None,
                   help="Optional root of GT videos for FVD (skipped if absent).")
    p.add_argument("--output_dir",
                   default=str(_EVAL_ROOT / "results" / "t2v"))
    p.add_argument("--sample_height", type=int, default=480)
    p.add_argument("--sample_width", type=int, default=832)
    p.add_argument("--video_length", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.prompts_json) as f:
        entries = json.load(f)  # [{"name": ..., "prompt": ...}, ...]

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
    )
    clip_model, clip_processor = load_clip(device=device)
    i3d_model = load_i3d(device=device) if args.gt_root else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_clips = []
    gt_clips = []
    per_video = []

    for entry in entries:
        name = entry["name"]
        prompt = entry["prompt"]
        print(f"\n── {name} ──")

        gen_path = str(out_dir / f"{name}.mp4")
        # T2V: no first frame or track conditioning → pass empty/None
        generator.generate(
            first_frame_path="",       # T2V mode: generator should handle empty path
            track_npz_path="",
            prompt=prompt,
            output_path=gen_path,
            seed=args.seed,
            fps=args.fps,
        )

        import cv2
        cap = cv2.VideoCapture(gen_path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        gen_arr = np.stack(frames, axis=0) if frames else np.zeros((1, args.sample_height, args.sample_width, 3), dtype=np.uint8)

        clip_s = compute_clip_score(gen_arr, prompt, clip_model, clip_processor, device)
        print(f"  CLIP={clip_s:.4f}")
        gen_clips.append(gen_arr)
        per_video.append({"name": name, "clip_score": clip_s})

        if args.gt_root:
            gt_path = Path(args.gt_root) / f"{name}.mp4"
            if gt_path.is_file():
                cap = cv2.VideoCapture(str(gt_path))
                gf = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    gf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
                gt_clips.append(np.stack(gf, axis=0))

    mean_clip = float(np.mean([m["clip_score"] for m in per_video if not np.isnan(m["clip_score"])]))
    fvd_val = compute_fvd(gt_clips, gen_clips, device, i3d_model) if gt_clips and gen_clips else float("nan")

    agg = {"clip_score": mean_clip, "fvd": fvd_val, "num_videos": len(per_video)}
    print(f"\nCLIP={mean_clip:.4f}  FVD={fvd_val:.2f}")

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    with open(out_dir / "per_video.json", "w") as f:
        json.dump(per_video, f, indent=2)
    print(f"[done] Results → {out_dir}")


if __name__ == "__main__":
    main()
