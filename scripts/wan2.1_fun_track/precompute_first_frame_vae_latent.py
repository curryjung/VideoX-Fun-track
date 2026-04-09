#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from diffusers.image_processor import VaeImageProcessor
from videox_fun.models import AutoencoderKLWan


def _load_records(meta_path: str) -> List[Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Metadata must be list: {meta_path}")
    return data


def _resolve_path(path: str, data_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(data_root, path))


def _read_first_frame_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _load_latent_shape(latent_path: str) -> Tuple[int, int, int, int]:
    latents = torch.load(latent_path, map_location="cpu")
    if not isinstance(latents, torch.Tensor):
        raise ValueError(f"Expected tensor latent file: {latent_path}")
    if latents.ndim == 5 and latents.shape[0] == 1:
        latents = latents[0]
    if latents.ndim != 4:
        raise ValueError(f"Unexpected latent shape {tuple(latents.shape)}: {latent_path}")
    c, f, h, w = latents.shape
    return int(c), int(f), int(h), int(w)


def _encode_masked_video_like_pipeline(masked_video: torch.Tensor, vae, device: torch.device) -> torch.Tensor:
    masked_video = masked_video.to(device=device, dtype=vae.dtype)
    chunks: List[torch.Tensor] = []
    micro_bs = 1
    for i in range(0, masked_video.shape[0], micro_bs):
        mb = masked_video[i : i + micro_bs]
        with torch.inference_mode():
            mb_latents = vae.encode(mb)[0].mode()
        chunks.append(mb_latents)
    return torch.cat(chunks, dim=0)


def _build_masked_video_latents(
    first_frame: torch.Tensor,
    vae,
    image_processor: VaeImageProcessor,
    mask_processor: VaeImageProcessor,
    device: torch.device,
    weight_dtype: torch.dtype,
    video_length: int,
    target_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    first_frame = first_frame.to(device=device, dtype=torch.float32)[None]  # [1,3,H,W]
    _, _, ff_h, ff_w = first_frame.shape

    video = torch.tile(first_frame.unsqueeze(2), [1, 1, video_length, 1, 1])
    mask_video = torch.zeros((1, 1, video_length, ff_h, ff_w), device=device, dtype=torch.float32)
    mask_video[:, :, 1:] = 255.0

    init_video = image_processor.preprocess(
        rearrange(video, "b c f h w -> (b f) c h w"),
        height=ff_h,
        width=ff_w,
    )
    init_video = init_video.to(dtype=torch.float32, device=device)
    init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)

    mask_condition = mask_processor.preprocess(
        rearrange(mask_video, "b c f h w -> (b f) c h w"),
        height=ff_h,
        width=ff_w,
    )
    mask_condition = mask_condition.to(dtype=torch.float32, device=device)
    mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

    masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
    masked_video_latents = _encode_masked_video_like_pipeline(
        masked_video=masked_video,
        vae=vae,
        device=device,
    ).to(dtype=weight_dtype)

    _, tgt_f, tgt_h, tgt_w = target_shape
    if masked_video_latents.shape[2:] != (tgt_f, tgt_h, tgt_w):
        masked_video_latents = F.interpolate(
            masked_video_latents,
            size=(tgt_f, tgt_h, tgt_w),
            mode="trilinear",
            align_corners=False,
        )
    return masked_video_latents[0].detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute first-frame VAE latent tensors for Wan2.1 track latent training.")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP",
    )
    parser.add_argument("--latent_file_key", type=str, default="latent_file_path")
    parser.add_argument("--first_frame_file_key", type=str, default="first_frame_file_path")
    parser.add_argument(
        "--first_frame_vae_latent_file_key",
        type=str,
        default="first_frame_vae_latent_file_path",
    )
    parser.add_argument("--output_filename", type=str, default="first_frame_vae_latent.pt")
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="For debugging. <=0 means all records.")
    parser.add_argument("--save_every", type=int, default=2000, help="Update metadata file every N processed samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    records = _load_records(args.metadata_path)
    if args.limit > 0:
        records = records[: args.limit]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for first-frame VAE latent precompute.")
    device = torch.device("cuda")

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    weight_dtype = dtype_map[args.mixed_precision]

    cfg = OmegaConf.load(args.config_path)
    vae_kwargs = OmegaConf.to_container(cfg["vae_kwargs"])
    if not isinstance(vae_kwargs, dict):
        raise ValueError("`vae_kwargs` must be a dict in config.")
    vae_subpath = str(vae_kwargs.get("vae_subpath", "vae"))
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, vae_subpath),
        additional_kwargs=vae_kwargs,
    ).to(device=device, dtype=weight_dtype).eval()

    image_processor = VaeImageProcessor(vae_scale_factor=vae.config.spatial_compression_ratio)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae.config.spatial_compression_ratio,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    written = 0
    skipped = 0
    failed = 0
    progress = tqdm(records, desc="precompute-first-frame-vae", unit="sample")
    for i, record in enumerate(progress, start=1):
        try:
            latent_rel = record.get(args.latent_file_key, record.get("file_path", ""))
            if latent_rel == "":
                skipped += 1
                continue
            latent_path = _resolve_path(latent_rel, data_root)
            if not os.path.isfile(latent_path):
                skipped += 1
                continue

            first_frame_rel = record.get(
                args.first_frame_file_key,
                os.path.join(os.path.dirname(latent_rel), "first_frame.png"),
            )
            first_frame_path = _resolve_path(first_frame_rel, data_root)
            if not os.path.isfile(first_frame_path):
                skipped += 1
                continue

            output_rel = record.get(
                args.first_frame_vae_latent_file_key,
                os.path.join(os.path.dirname(latent_rel), args.output_filename),
            )
            output_path = _resolve_path(output_rel, data_root)
            if (not args.overwrite) and os.path.isfile(output_path):
                if args.first_frame_vae_latent_file_key not in record:
                    record[args.first_frame_vae_latent_file_key] = output_rel
                skipped += 1
                continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            target_shape = _load_latent_shape(latent_path)
            _, latent_f, _, _ = target_shape
            video_length = int(1 + max(latent_f - 1, 0) * 4)
            first_frame_tensor = _read_first_frame_tensor(first_frame_path)
            first_frame_vae_latent = _build_masked_video_latents(
                first_frame=first_frame_tensor,
                vae=vae,
                image_processor=image_processor,
                mask_processor=mask_processor,
                device=device,
                weight_dtype=weight_dtype,
                video_length=video_length,
                target_shape=target_shape,
            )
            torch.save(first_frame_vae_latent, output_path)
            record[args.first_frame_vae_latent_file_key] = output_rel
            written += 1
        except Exception:
            failed += 1

        if args.save_every > 0 and i % args.save_every == 0:
            with open(args.metadata_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        progress.set_postfix(written=written, skipped=skipped, failed=failed)

    with open(args.metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(
        f"[done] written={written} skipped={skipped} failed={failed} "
        f"metadata={args.metadata_path}"
    )


if __name__ == "__main__":
    main()
