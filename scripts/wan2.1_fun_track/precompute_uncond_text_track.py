#!/usr/bin/env python3
"""Write null-condition T5 npz (same keys as text_feature_wan_t5.npz) for wan2.1_fun_track training."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from videox_fun.models import WanT5EncoderModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--output_npz", type=str, required=True)
    p.add_argument("--tokenizer_max_length", type=int, default=512)
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Encode this string as unconditional (default '' matches train_track caption drop).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Where to run T5: "cuda" or "cpu".',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    weight_dtype = torch.float32
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        weight_dtype = torch.bfloat16

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    text_encoder.to(device=device, dtype=weight_dtype)

    with torch.no_grad():
        batch_enc = tokenizer(
            [args.prompt],
            padding="max_length",
            max_length=args.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = batch_enc.input_ids.to(device)
        attn = batch_enc.attention_mask.to(device)
        hidden = text_encoder(input_ids, attention_mask=attn)[0]
        valid_len = int(attn[0].sum().item())
        if valid_len <= 0:
            valid_len = int(hidden.shape[1])
        pe = hidden[0, :valid_len].float().cpu().numpy()
        am = attn[0, :valid_len].long().cpu().numpy()

    out_abs = os.path.abspath(args.output_npz)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(out_abs, prompt_embeds=pe, attention_mask=am)
    print(
        f"Wrote {out_abs}  prompt_embeds={pe.shape}  attention_mask={am.shape}  "
        f"dtype={pe.dtype}  prompt_repr={args.prompt!r}"
    )


if __name__ == "__main__":
    main()
