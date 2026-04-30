"""
Video generation wrapper for motion-transfer evaluation.

Loads the Wan2.1-Fun-Track I2V pipeline once and exposes a simple
generate() interface used by benchmarks.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

_EVAL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _EVAL_ROOT.parent
for _p in [str(_EVAL_ROOT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from videox_fun.models import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModelTrack,
)
from videox_fun.pipeline import WanFunInpaintPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid


def _resolve_transformer_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    for name in [
        "pytorch_model.bin", "model.safetensors",
        "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors",
    ]:
        c = os.path.join(path, name)
        if os.path.isfile(c):
            return c
    hits = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith((".safetensors", ".bin")):
                if not any(k in name.lower() for k in ("optimizer", "scheduler")):
                    hits.append(os.path.join(root, name))
    if not hits:
        raise FileNotFoundError(f"No weight file found under: {path}")
    return sorted(hits)[0]


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise ValueError(f"Unsupported checkpoint type: {type(obj)}")


class VideoGenerator:
    """
    Wraps the Wan2.1-Fun-Track I2V pipeline for repeated inference.

    Load once at init, call generate() per video.
    """

    def __init__(
        self,
        model_name: str,
        config_path: str,
        transformer_checkpoint_path: Optional[str] = None,
        mixed_precision: str = "bf16",
        device: str = "cuda",
        sample_height: int = 480,
        sample_width: int = 832,
        video_length: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        guidance_mode: str = "cfg",
        sampler_name: str = "Flow",
        shift: float = 3.0,
        normalize_track: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.video_length = video_length
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_mode = guidance_mode
        self.shift = shift
        self.normalize_track = normalize_track

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        self.weight_dtype = dtype_map[mixed_precision]

        config = OmegaConf.load(config_path)

        transformer_subpath = config["transformer_additional_kwargs"].get(
            "transformer_subpath", "transformer"
        )
        transformer = WanTransformer3DModelTrack.from_pretrained(
            os.path.join(model_name, transformer_subpath),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )

        if transformer_checkpoint_path:
            ckpt_file = _resolve_transformer_checkpoint(transformer_checkpoint_path)
            print(f"[VideoGenerator] Loading checkpoint: {ckpt_file}")
            sd = _load_state_dict(ckpt_file)
            cleaned = {}
            for k, v in sd.items():
                for prefix in ("module.", "_orig_mod.", "transformer3d_track.", "transformer."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                cleaned[k] = v
            missing, unexpected = transformer.load_state_dict(cleaned, strict=False)
            print(f"[VideoGenerator] missing={len(missing)}, unexpected={len(unexpected)}")

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(self.weight_dtype)
        self.temporal_compression_ratio = vae.config.temporal_compression_ratio

        tok_subpath = config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
        te_subpath = config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_name, tok_subpath))
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, te_subpath),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).eval()

        ie_subpath = config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")
        clip_image_encoder = (
            CLIPModel.from_pretrained(os.path.join(model_name, ie_subpath))
            .to(self.weight_dtype)
            .eval()
        )

        scheduler_cls = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }[sampler_name]
        scheduler_cfg = OmegaConf.to_container(config["scheduler_kwargs"])
        if sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
            scheduler_cfg["shift"] = 1
        scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_cfg))

        self.pipeline = WanFunInpaintPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        ).to(device=self.device)

        print("[VideoGenerator] Pipeline ready.")

    def generate(
        self,
        first_frame_path: str,
        track_npz_path: str,
        prompt: str,
        output_path: str,
        seed: int = 42,
        fps: int = 16,
        negative_prompt: str = "worst quality, low quality, blurry, static frame",
        video_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a video from a first frame, track conditioning, and text prompt.

        Args:
            video_length: override the default model video length (frames). Useful
                          when the GT video is shorter than the model's default temporal
                          length — pass T_gt so tracks and generated frames match.
                          Must be ≥ 1; the actual generated length is rounded down to
                          the nearest valid latent length.

        Returns:
            sample: (1, 3, T, H, W) float32 tensor in [0, 1].
        """
        target_len = video_length if video_length is not None else self.video_length
        video_length = (
            int(
                (target_len - 1)
                // self.temporal_compression_ratio
                * self.temporal_compression_ratio
            )
            + 1
        )

        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            first_frame_path,
            None,
            video_length=video_length,
            sample_size=[self.sample_height, self.sample_width],
        )

        track_condition = self._build_track_condition(track_npz_path, video_length)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            sample = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=video_length,
                height=self.sample_height,
                width=self.sample_width,
                generator=generator,
                guidance_mode=self.guidance_mode,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                shift=self.shift,
                track_condition=track_condition,
            ).videos

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_videos_grid(sample, output_path, fps=fps)
        return sample

    def _build_track_condition(
        self, track_npz_path: str, video_length: int
    ) -> Optional[dict]:
        if not track_npz_path or not os.path.isfile(track_npz_path):
            return None

        data = np.load(track_npz_path)
        tracks = data.get("tracks_compressed", data.get("tracks")).astype(np.float32)
        if "visibility_compressed" in data:
            visibility = data["visibility_compressed"].astype(np.float32)
        elif "visibility" in data:
            visibility = data["visibility"].astype(np.float32)
        else:
            visibility = np.ones(tracks.shape[:2], dtype=np.float32)

        if tracks.ndim == 4 and tracks.shape[0] == 1:
            tracks = tracks[0]
        if visibility.ndim == 3 and visibility.shape[0] == 1:
            visibility = visibility[0]
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility.squeeze(-1)

        if tracks.shape[0] > video_length:
            tracks = tracks[:video_length]
            visibility = visibility[:video_length]

        tracks_t = torch.as_tensor(tracks, dtype=torch.float32, device=self.device)
        visibility_t = torch.as_tensor(visibility, dtype=torch.float32, device=self.device)

        if self.normalize_track:
            tracks_t[..., 0] /= float(self.sample_width)
            tracks_t[..., 1] /= float(self.sample_height)

        N = tracks_t.shape[1]
        return {
            "tracks": tracks_t.unsqueeze(0),
            "visibility": visibility_t.unsqueeze(0),
            "point_mask": torch.ones((1, N), dtype=torch.bool, device=self.device),
            "point_ids": torch.arange(N, dtype=torch.long, device=self.device).unsqueeze(0),
            "is_normalized": torch.tensor([self.normalize_track], dtype=torch.bool, device=self.device),
            "track_resolution": torch.tensor(
                [[float(self.sample_width), float(self.sample_height)]],
                dtype=torch.float32,
                device=self.device,
            ),
        }
