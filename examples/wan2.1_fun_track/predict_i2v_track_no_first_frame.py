import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.models import (  # noqa: E402
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModelTrack,
)
from videox_fun.pipeline import WanFunInpaintPipeline  # noqa: E402
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: E402
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from videox_fun.utils.utils import (  # noqa: E402
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)


def _load_state_dict_from_path(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_obj = torch.load(path, map_location="cpu")
        if isinstance(state_obj, dict):
            if "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
                state_dict = state_obj["state_dict"]
            elif "model" in state_obj and isinstance(state_obj["model"], dict):
                state_dict = state_obj["model"]
            else:
                state_dict = state_obj
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(state_obj)}")
    return state_dict


def _resolve_transformer_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    ]
    for name in candidates:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate

    recursive_hits = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith((".safetensors", ".bin")):
                lowered = name.lower()
                if "optimizer" in lowered or "scheduler" in lowered:
                    continue
                recursive_hits.append(os.path.join(root, name))
    if not recursive_hits:
        raise FileNotFoundError(
            f"No transformer weight file found under checkpoint directory: {path}"
        )
    recursive_hits.sort()
    return recursive_hits[0]


def _load_text_feature_npz(
    text_feature_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    data = np.load(text_feature_path, allow_pickle=True)
    if "prompt_embeds" not in data:
        raise KeyError(f"`prompt_embeds` key not found in: {text_feature_path}")
    prompt_embeds = torch.as_tensor(data["prompt_embeds"], dtype=dtype, device=device)
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.ndim != 3:
        raise ValueError(
            f"prompt_embeds must be [L,D] or [B,L,D], got {tuple(prompt_embeds.shape)}"
        )
    return prompt_embeds


def _read_metadata_records(metadata_path: str) -> List[Dict]:
    lower = metadata_path.lower()
    if lower.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"JSON metadata must be a list: {metadata_path}")
        return records

    if lower.endswith(".jsonl"):
        records: List[Dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                records.append(json.loads(line))
        return records

    if lower.endswith(".csv"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported metadata format: {metadata_path}")


def _select_metadata_row(
    metadata_path: str,
    sample_index: int,
    random_sample: bool,
) -> Tuple[Dict, int, int]:
    records = _read_metadata_records(metadata_path)
    if len(records) == 0:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    if random_sample:
        selected_index = random.randint(0, len(records) - 1)
    else:
        selected_index = int(sample_index)
        if selected_index < 0:
            selected_index += len(records)
        if selected_index < 0 or selected_index >= len(records):
            raise IndexError(
                f"sample_index={sample_index} out of range for metadata length={len(records)}"
            )
    return records[selected_index], selected_index, len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.1 Fun Track ablation: i2v inference without first frame and without track condition."
    )
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
        "--transformer_checkpoint_path",
        type=str,
        default=None,
        help="Path to fine-tuned transformer checkpoint file or checkpoint directory.",
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, low quality, blurry, static frame, bad anatomy",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional metadata path (json/jsonl/csv). If set, one sample is selected for prompt loading.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Metadata row index to use when --metadata_path is set.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Use random metadata row (overrides --sample_index).",
    )
    parser.add_argument(
        "--use_prompt_from_metadata",
        action="store_true",
        help="Use selected metadata row `text` as prompt.",
    )
    parser.add_argument(
        "--text_feature_path",
        type=str,
        default=None,
        help="Optional precomputed text feature npz with `prompt_embeds`.",
    )
    parser.add_argument(
        "--negative_text_feature_path",
        type=str,
        default=None,
        help="Optional precomputed negative text feature npz with `prompt_embeds`.",
    )
    parser.add_argument("--sample_height", type=int, default=480)
    parser.add_argument("--sample_width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="cfg",
        choices=["cfg", "text_only"],
        help="No-track mode supports cfg and text_only guidance.",
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--text_guidance_weight",
        type=float,
        default=3.0,
        help="Text guidance weight used when guidance_mode=text_only.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="Flow",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="samples/wan-videos-fun-i2v-track-no-first-frame",
    )
    parser.add_argument(
        "--output_name_suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.1 inference.")
    device = torch.device("cuda")

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    config = OmegaConf.load(args.config_path)

    transformer = WanTransformer3DModelTrack.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if args.transformer_checkpoint_path:
        resolved_ckpt = _resolve_transformer_checkpoint(args.transformer_checkpoint_path)
        print(f"[info] loading finetuned transformer checkpoint: {resolved_ckpt}")
        state_dict = _load_state_dict_from_path(resolved_ckpt)
        cleaned_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod.") :]
            if new_key.startswith("transformer3d_track."):
                new_key = new_key[len("transformer3d_track.") :]
            if new_key.startswith("transformer."):
                new_key = new_key[len("transformer.") :]
            cleaned_state_dict[new_key] = value
        missing, unexpected = transformer.load_state_dict(cleaned_state_dict, strict=False)
        print(
            f"[info] checkpoint load done: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            args.model_name,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        ),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
        ),
    ).to(weight_dtype).eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    scheduler_cfg = OmegaConf.to_container(config["scheduler_kwargs"])
    if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        scheduler_cfg["shift"] = 1
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_cfg))

    pipeline = WanFunInpaintPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    ).to(device=device)

    selected_metadata = None
    selected_index = None
    metadata_total = None
    if args.metadata_path is not None and args.metadata_path != "":
        selected_metadata, selected_index, metadata_total = _select_metadata_row(
            metadata_path=args.metadata_path,
            sample_index=args.sample_index,
            random_sample=args.random_sample,
        )
        print(
            f"[info] metadata sample selected: index={selected_index}, total={metadata_total}"
        )
        if args.use_prompt_from_metadata:
            meta_prompt = str(selected_metadata.get("text", "")).strip()
            if meta_prompt != "":
                args.prompt = meta_prompt
                print("[info] prompt loaded from metadata `text`.")

    if (args.text_feature_path is None or args.text_feature_path == "") and args.prompt.strip() == "":
        raise ValueError(
            "prompt is empty. Pass --prompt or set --use_prompt_from_metadata with metadata row text."
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    video_length = (
        int(
            (args.video_length - 1)
            // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio
        )
        + 1
        if args.video_length != 1
        else 1
    )

    # First-frame-free setting:
    # pass None to generate empty inpaint inputs (all-mask, zero latent context).
    input_video, input_video_mask, _ = get_image_to_video_latent(
        None,
        None,
        video_length=video_length,
        sample_size=[args.sample_height, args.sample_width],
    )

    prompt_embeds = None
    negative_prompt_embeds = None
    if args.text_feature_path is not None and args.text_feature_path != "":
        prompt_embeds = _load_text_feature_npz(
            text_feature_path=args.text_feature_path,
            device=device,
            dtype=weight_dtype,
        )
        print(
            "[info] loaded external text features: "
            f"prompt_embeds={tuple(prompt_embeds.shape)}"
        )

    if args.negative_text_feature_path is not None and args.negative_text_feature_path != "":
        negative_prompt_embeds = _load_text_feature_npz(
            text_feature_path=args.negative_text_feature_path,
            device=device,
            dtype=weight_dtype,
        )
        print(
            "[info] loaded external negative text features: "
            f"negative_prompt_embeds={tuple(negative_prompt_embeds.shape)}"
        )
    elif prompt_embeds is not None:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError(
                "negative prompt embeds shape mismatch: "
                f"{tuple(negative_prompt_embeds.shape)} vs {tuple(prompt_embeds.shape)}"
            )

    with torch.no_grad():
        sample = pipeline(
            prompt=None if prompt_embeds is not None else args.prompt,
            negative_prompt=None if negative_prompt_embeds is not None else args.negative_prompt,
            num_frames=video_length,
            height=args.sample_height,
            width=args.sample_width,
            generator=generator,
            guidance_mode=args.guidance_mode,
            guidance_scale=args.guidance_scale,
            text_guidance_weight=args.text_guidance_weight,
            motion_guidance_weight=0.0,
            num_inference_steps=args.num_inference_steps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=None,
            clip_feature=None,
            shift=args.shift,
            track_condition=None,
        ).videos

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(args.output_name_suffix).strip()
    suffix_part = f"_{suffix}" if suffix != "" else ""
    output_path = os.path.join(
        args.save_dir,
        f"track_i2v_no_first_frame_no_track{suffix_part}_{timestamp}.mp4",
    )
    save_videos_grid(sample, output_path, fps=args.fps)
    print(f"[done] saved video: {output_path}")

    run_info = {
        "timestamp": timestamp,
        "output_path": output_path,
        "first_frame_input": None,
        "track_condition": None,
        "metadata_path": args.metadata_path,
        "selected_metadata_index": selected_index,
        "metadata_total": metadata_total,
        "guidance_mode": args.guidance_mode,
        "seed": args.seed,
        "video_length": int(video_length),
        "sample_height": int(args.sample_height),
        "sample_width": int(args.sample_width),
    }
    if selected_metadata is not None and isinstance(selected_metadata, dict):
        run_info["metadata_file_path"] = str(selected_metadata.get("file_path", ""))
        run_info["metadata_track_file_path"] = str(selected_metadata.get("track_file_path", ""))
    run_info_path = os.path.join(
        args.save_dir,
        f"track_i2v_no_first_frame_no_track{suffix_part}_{timestamp}.json",
    )
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"[done] saved run info: {run_info_path}")


if __name__ == "__main__":
    main()
