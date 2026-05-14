from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# _TORCHVISION_LIB = None


# def _ensure_torchvision_nms_stub() -> None:
#     """Avoid torchvision import failures when the optional NMS op is unavailable."""
#     global _TORCHVISION_LIB
#     try:
#         _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
#         _TORCHVISION_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
#     except RuntimeError:
#         pass


# _ensure_torchvision_nms_stub()

try:
    import transformers.utils.import_utils as _transformers_import_utils
    import transformers.utils as _transformers_utils

    for _module in (_transformers_import_utils, _transformers_utils):
        _module.is_flash_attn_2_available = lambda: False
        _module.is_flash_attn_greater_or_equal_2_10 = lambda: False
        _module.is_flash_attn_greater_or_equal = lambda _: False
except Exception:
    pass

try:
    import diffusers.utils.import_utils as _diffusers_import_utils
    import diffusers.utils as _diffusers_utils

    _diffusers_import_utils._flash_attn_available = False
    _diffusers_import_utils._flash_attn_3_available = False
    _diffusers_import_utils.is_flash_attn_available = lambda: False
    _diffusers_import_utils.is_flash_attn_3_available = lambda: False
    _diffusers_import_utils.is_flash_attn_version = lambda *_args, **_kwargs: False
    _diffusers_utils.is_flash_attn_available = lambda: False
    _diffusers_utils.is_flash_attn_3_available = lambda: False
    _diffusers_utils.is_flash_attn_version = lambda *_args, **_kwargs: False
except Exception:
    pass

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from videox_fun.models import (  # noqa: E402
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
    WanTransformer3DModelTrack,
)
from videox_fun.pipeline import WanFunInpaintPipeline  # noqa: E402
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: E402
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid  # noqa: E402


def _resolve_transformer_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    for name in (
        "pytorch_model.bin",
        "model.safetensors",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    ):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate

    hits: list[str] = []
    for root, _, files in os.walk(path):
        for name in files:
            lower = name.lower()
            if lower.endswith((".safetensors", ".bin")) and not any(
                token in lower for token in ("optimizer", "scheduler")
            ):
                hits.append(os.path.join(root, name))
    if not hits:
        raise FileNotFoundError(f"No transformer weight file found under: {path}")
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
    raise ValueError(f"Unsupported checkpoint object type: {type(obj)}")


def _load_text_feature_npz(
    text_feature_path: str,
    *,
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


def _load_predict_module():
    path = _PROJECT_ROOT / "examples" / "wan2.1_fun_track" / "predict_i2v_track.py"
    spec = importlib.util.spec_from_file_location("_wan_track_predict_i2v_track", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predict module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackConditionedVideoGenerator:
    def __init__(
        self,
        *,
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
        guidance_mode: str = "motion_only",
        text_guidance_weight: float = 0.0,
        motion_guidance_weight: float = 3.5,
        negative_text_feature_path: Optional[str] = None,
        sampler_name: str = "Flow",
        shift: float = 3.0,
        normalize_track: bool = True,
        track_normalize_height: int = 480,
        track_normalize_width: int = 832,
        track_latent_scale: float = 1.0,
        track_latent_first_frame_scale: Optional[float] = None,
        track_latent_rest_frame_scale: Optional[float] = None,
        track_head_hidden_dim: Optional[int] = None,
        track_condition_mode: str = "track_head",
        wan_move_temporal_stride: int = 0,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_height = int(sample_height)
        self.sample_width = int(sample_width)
        self.video_length = int(video_length)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.guidance_mode = str(guidance_mode)
        self.text_guidance_weight = float(text_guidance_weight)
        self.motion_guidance_weight = float(motion_guidance_weight)
        self.shift = float(shift)
        self.normalize_track = bool(normalize_track)
        self.track_normalize_height = int(track_normalize_height)
        self.track_normalize_width = int(track_normalize_width)

        if not np.isfinite(float(track_latent_scale)) or float(track_latent_scale) < 0.0:
            raise ValueError("track_latent_scale must be a finite non-negative value.")
        split_track_latent_scale = (
            track_latent_first_frame_scale is not None
            or track_latent_rest_frame_scale is not None
        )
        if split_track_latent_scale:
            if track_latent_first_frame_scale is None:
                track_latent_first_frame_scale = float(track_latent_scale)
            if track_latent_rest_frame_scale is None:
                track_latent_rest_frame_scale = float(track_latent_scale)
            for name, value in (
                ("track_latent_first_frame_scale", track_latent_first_frame_scale),
                ("track_latent_rest_frame_scale", track_latent_rest_frame_scale),
            ):
                if not np.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"{name} must be a finite non-negative value.")
        self.track_latent_first_frame_scale = (
            None if track_latent_first_frame_scale is None else float(track_latent_first_frame_scale)
        )
        self.track_latent_rest_frame_scale = (
            None if track_latent_rest_frame_scale is None else float(track_latent_rest_frame_scale)
        )

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if mixed_precision not in dtype_map:
            raise ValueError(f"Unsupported mixed precision: {mixed_precision}")
        weight_dtype = dtype_map[mixed_precision]
        self.weight_dtype = weight_dtype
        self.negative_text_feature_path = negative_text_feature_path
        self.negative_prompt_embeds = None
        if negative_text_feature_path:
            self.negative_prompt_embeds = _load_text_feature_npz(
                negative_text_feature_path,
                device=self.device,
                dtype=weight_dtype,
            )
            print(
                "[generator] Loaded external negative text features: "
                f"{tuple(self.negative_prompt_embeds.shape)}"
            )

        config = OmegaConf.load(config_path)
        transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
        mode = str(track_condition_mode).strip().lower()
        if mode == "track_head":
            if track_head_hidden_dim is not None:
                if int(track_head_hidden_dim) <= 0:
                    raise ValueError("track_head_hidden_dim must be > 0 when provided.")
                transformer_kwargs["track_head_hidden_dim"] = int(track_head_hidden_dim)
            transformer_kwargs["track_latent_scale"] = (
                float(self.track_latent_rest_frame_scale)
                if split_track_latent_scale
                else float(track_latent_scale)
            )
            transformer_cls = WanTransformer3DModelTrack
        elif mode == "wan_move":
            transformer_cls = WanTransformer3DModel
        else:
            raise ValueError(f"Unsupported track_condition_mode: {track_condition_mode}")

        transformer_subpath = config["transformer_additional_kwargs"].get(
            "transformer_subpath", "transformer"
        )
        transformer = transformer_cls.from_pretrained(
            os.path.join(model_name, transformer_subpath),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        if transformer_checkpoint_path:
            ckpt_file = _resolve_transformer_checkpoint(transformer_checkpoint_path)
            print(f"[generator] Loading transformer checkpoint: {ckpt_file}")
            state_dict = _load_state_dict(ckpt_file)
            cleaned = {}
            for key, value in state_dict.items():
                new_key = key
                for prefix in ("module.", "_orig_mod.", "transformer3d_track.", "transformer."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                cleaned[new_key] = value
            missing, unexpected = transformer.load_state_dict(cleaned, strict=False)
            print(f"[generator] checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)
        self.temporal_compression_ratio = int(getattr(vae.config, "temporal_compression_ratio", 4))

        if mode == "wan_move":
            stride = int(wan_move_temporal_stride)
            if stride <= 0:
                stride = self.temporal_compression_ratio
            predict_module = _load_predict_module()
            predict_module._attach_wan_move_forward_adapter(
                transformer=transformer,
                latent_channels=int(getattr(vae.config, "latent_channels", 16)),
                temporal_stride=stride,
            )
            print(f"[generator] Wan-Move adapter enabled with temporal_stride={stride}")

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(
                model_name,
                config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            ),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()
        clip_image_encoder = (
            CLIPModel.from_pretrained(
                os.path.join(
                    model_name,
                    config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
                )
            )
            .to(weight_dtype)
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
        print("[generator] Pipeline ready.")

    def round_video_length(self, video_length: int) -> int:
        if int(video_length) == 1:
            return 1
        return int((int(video_length) - 1) // self.temporal_compression_ratio) * (
            self.temporal_compression_ratio
        ) + 1

    def build_track_condition(
        self,
        tracks: np.ndarray,
        visibility: np.ndarray,
        *,
        video_length: int,
        point_ids: Optional[np.ndarray] = None,
    ) -> dict:
        tracks = np.asarray(tracks, dtype=np.float32)
        visibility = np.asarray(visibility, dtype=np.float32)
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"Expected tracks (T, N, 2), got {tracks.shape}")
        if visibility.shape != tracks.shape[:2]:
            raise ValueError(
                f"Track/visibility mismatch: tracks={tracks.shape}, visibility={visibility.shape}"
            )

        length = int(video_length)
        if tracks.shape[0] > length:
            tracks = tracks[:length]
            visibility = visibility[:length]

        tracks_t = torch.as_tensor(tracks, dtype=torch.float32, device=self.device)
        visibility_t = torch.as_tensor(visibility, dtype=torch.float32, device=self.device)
        if self.normalize_track:
            tracks_t = tracks_t.clone()
            tracks_t[..., 0] /= float(self.track_normalize_width)
            tracks_t[..., 1] /= float(self.track_normalize_height)

        num_points = tracks_t.shape[1]
        if point_ids is None:
            point_ids_t = torch.arange(num_points, dtype=torch.long, device=self.device)
        else:
            point_ids_t = torch.as_tensor(point_ids, dtype=torch.long, device=self.device)
            if point_ids_t.numel() != num_points:
                raise ValueError(
                    f"point_ids length {point_ids_t.numel()} does not match num_points={num_points}"
                )

        return {
            "tracks": tracks_t.unsqueeze(0),
            "visibility": visibility_t.unsqueeze(0),
            "point_mask": torch.ones((1, num_points), dtype=torch.bool, device=self.device),
            "point_ids": point_ids_t.unsqueeze(0),
            "is_normalized": torch.tensor([self.normalize_track], dtype=torch.bool, device=self.device),
            "track_resolution": torch.tensor(
                [[float(self.track_normalize_width), float(self.track_normalize_height)]],
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def generate(
        self,
        *,
        first_frame_path: str,
        tracks: np.ndarray,
        visibility: np.ndarray,
        output_path: Optional[str] = None,
        prompt: str,
        negative_prompt: str,
        seed: int,
        fps: int,
        video_length: Optional[int] = None,
        point_ids: Optional[np.ndarray] = None,
        save_video: bool = True,
    ) -> tuple[torch.Tensor, int]:
        target_len = int(video_length if video_length is not None else self.video_length)
        actual_len = self.round_video_length(target_len)
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            first_frame_path,
            None,
            video_length=actual_len,
            sample_size=[self.sample_height, self.sample_width],
        )
        track_condition = self.build_track_condition(
            tracks,
            visibility,
            video_length=actual_len,
            point_ids=point_ids,
        )
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        old_first_scale = os.environ.get("TRACK_LATENT_FIRST_FRAME_SCALE")
        old_rest_scale = os.environ.get("TRACK_LATENT_REST_FRAME_SCALE")
        try:
            if self.track_latent_first_frame_scale is not None:
                os.environ["TRACK_LATENT_FIRST_FRAME_SCALE"] = str(
                    self.track_latent_first_frame_scale
                )
            if self.track_latent_rest_frame_scale is not None:
                os.environ["TRACK_LATENT_REST_FRAME_SCALE"] = str(
                    self.track_latent_rest_frame_scale
                )

            with torch.no_grad():
                sample = self.pipeline(
                    prompt=prompt,
                    negative_prompt=None if self.negative_prompt_embeds is not None else negative_prompt,
                    num_frames=actual_len,
                    height=self.sample_height,
                    width=self.sample_width,
                    generator=generator,
                    guidance_mode=self.guidance_mode,
                    guidance_scale=self.guidance_scale,
                    text_guidance_weight=self.text_guidance_weight,
                    motion_guidance_weight=self.motion_guidance_weight,
                    num_inference_steps=self.num_inference_steps,
                    negative_prompt_embeds=self.negative_prompt_embeds,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    shift=self.shift,
                    track_condition=track_condition,
                ).videos
        finally:
            if old_first_scale is None:
                os.environ.pop("TRACK_LATENT_FIRST_FRAME_SCALE", None)
            else:
                os.environ["TRACK_LATENT_FIRST_FRAME_SCALE"] = old_first_scale
            if old_rest_scale is None:
                os.environ.pop("TRACK_LATENT_REST_FRAME_SCALE", None)
            else:
                os.environ["TRACK_LATENT_REST_FRAME_SCALE"] = old_rest_scale

        if save_video:
            if output_path is None:
                raise ValueError("output_path must be provided when save_video=True.")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            save_videos_grid(sample, output_path, fps=int(fps))
        return sample, actual_len
