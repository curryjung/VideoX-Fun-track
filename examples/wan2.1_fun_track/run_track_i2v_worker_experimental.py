#!/usr/bin/env python
"""Experimental persistent worker for Track Builder UI jobs.

This keeps the Wan track model on GPU and consumes filesystem-backed jobs from
asset/track_builder_jobs. It intentionally leaves predict_i2v_track.py untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track_builder_ui.backend.app.job_runner import DEFAULT_CKPT, DEFAULT_EXP_NAME  # noqa: E402
from track_builder_ui.backend.app.job_store import JobStore  # noqa: E402
from track_builder_ui.backend.app.schemas import JobRecord  # noqa: E402


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _resolve_checkpoint(repo_root: Path) -> str:
    explicit = os.environ.get("TRACK_BUILDER_TRANSFORMER_CHECKPOINT_PATH", "").strip()
    if explicit:
        return explicit
    exp_name = os.environ.get("TRACK_BUILDER_EXP_NAME", DEFAULT_EXP_NAME).strip() or DEFAULT_EXP_NAME
    ckpt = os.environ.get("TRACK_BUILDER_CKPT", DEFAULT_CKPT).strip() or DEFAULT_CKPT
    return str(repo_root / "checkpoints" / exp_name / f"checkpoint-{ckpt}")


def _load_predict_module(repo_root: Path) -> Any:
    module_path = repo_root / "examples" / "wan2.1_fun_track" / "predict_i2v_track.py"
    spec = importlib.util.spec_from_file_location(
        "wan_track_predict_i2v_track_experimental",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load predict module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental persistent Track Builder worker.")
    parser.add_argument(
        "--jobs_root",
        type=str,
        default=_env("TRACK_BUILDER_JOBS_ROOT", str(REPO_ROOT / "asset" / "track_builder_jobs")),
    )
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("TRACK_BUILDER_CUDA_VISIBLE_DEVICES", "6")),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=_env("TRACK_BUILDER_MODEL_NAME", "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=_env("TRACK_BUILDER_CONFIG_PATH", "config/wan2.1/wan_civitai.yaml"),
    )
    parser.add_argument(
        "--transformer_checkpoint_path",
        type=str,
        default=_resolve_checkpoint(REPO_ROOT),
    )
    parser.add_argument("--mixed_precision", type=str, default=_env("TRACK_BUILDER_MIXED_PRECISION", "bf16"))
    parser.add_argument("--sampler_name", type=str, default=_env("TRACK_BUILDER_SAMPLER_NAME", "Flow"))
    parser.add_argument("--shift", type=float, default=float(_env("TRACK_BUILDER_SHIFT", "3.0")))
    parser.add_argument("--sample_height", type=int, default=int(_env("TRACK_BUILDER_SAMPLE_HEIGHT", "480")))
    parser.add_argument("--sample_width", type=int, default=int(_env("TRACK_BUILDER_SAMPLE_WIDTH", "832")))
    parser.add_argument("--video_length", type=int, default=int(_env("TRACK_BUILDER_VIDEO_LENGTH", "81")))
    parser.add_argument("--fps", type=int, default=int(_env("TRACK_BUILDER_FPS", "16")))
    parser.add_argument("--num_inference_steps", type=int, default=int(_env("TRACK_BUILDER_NUM_INFERENCE_STEPS", "50")))
    parser.add_argument("--guidance_scale", type=float, default=float(_env("TRACK_BUILDER_GUIDANCE_SCALE", "6.0")))
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_env("TRACK_BUILDER_NEGATIVE_PROMPT", "worst quality, low quality, blurry, static frame"),
    )
    parser.add_argument(
        "--default_uncond_text_npz",
        type=str,
        default=_env("TRACK_BUILDER_DEFAULT_UNCOND_TEXT_NPZ", str(REPO_ROOT / "asset" / "t5_uncond_empty_prompt.npz")),
    )
    parser.add_argument(
        "--track_head_hidden_dim",
        type=int,
        default=int(_env("TRACK_BUILDER_TRACK_HEAD_HIDDEN_DIM", "64")),
    )
    parser.add_argument(
        "--track_latent_scale",
        type=float,
        default=float(_env("TRACK_BUILDER_TRACK_LATENT_SCALE", "4.0")),
    )
    parser.add_argument("--track_max_points", type=int, default=int(_env("TRACK_BUILDER_TRACK_MAX_POINTS", "2000")))
    parser.add_argument(
        "--track_point_sample_mode",
        type=str,
        default=_env("TRACK_BUILDER_TRACK_POINT_SAMPLE_MODE", "random"),
        choices=["uniform", "random"],
    )
    parser.add_argument(
        "--track_sort_selected_indices",
        type=_str2bool,
        default=_str2bool(_env("TRACK_BUILDER_TRACK_SORT_SELECTED_INDICES", "false")),
    )
    parser.add_argument(
        "--track_point_id_mode",
        type=str,
        default=_env("TRACK_BUILDER_TRACK_POINT_ID_MODE", "local"),
        choices=["original", "local"],
    )
    parser.add_argument("--track_normalize", type=_str2bool, default=True)
    parser.add_argument("--track_normalize_height", type=int, default=480)
    parser.add_argument("--track_normalize_width", type=int, default=832)
    parser.add_argument("--overlay_linewidth", type=int, default=int(_env("TRACK_BUILDER_OVERLAY_LINEWIDTH", "1")))
    parser.add_argument("--overlay_trace_frames", type=int, default=int(_env("TRACK_BUILDER_OVERLAY_TRACE_FRAMES", "8")))
    parser.add_argument("--overlay_pad_value", type=int, default=int(_env("TRACK_BUILDER_OVERLAY_PAD_VALUE", "0")))
    parser.add_argument(
        "--cotracker_root",
        type=str,
        default=_env("TRACK_BUILDER_COTRACKER_ROOT", "/data/project-vilab/jaeseok/co-tracker"),
    )
    parser.add_argument("--debug_track_condition", action="store_true")
    parser.add_argument("--track_analysis", action="store_true")
    parser.add_argument("--zero_clip_context", action="store_true")
    return parser.parse_args()


class TrackI2VPersistentRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.predict = _load_predict_module(REPO_ROOT)
        self.torch = self.predict.torch
        self.np = self.predict.np

        if not self.np.isfinite(float(args.track_latent_scale)):
            raise ValueError("--track_latent_scale must be finite.")
        if float(args.track_latent_scale) < 0.0:
            raise ValueError("--track_latent_scale must be >= 0.")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is required for persistent Wan2.1 inference worker.")

        self.device = self.torch.device("cuda")
        if args.mixed_precision == "fp16":
            self.weight_dtype = self.torch.float16
        elif args.mixed_precision == "bf16":
            self.weight_dtype = self.torch.bfloat16
        elif args.mixed_precision == "fp32":
            self.weight_dtype = self.torch.float32
        else:
            raise ValueError(f"Unsupported mixed_precision: {args.mixed_precision}")

        print("[worker] loading config/model/checkpoint once...")
        self.config = self.predict.OmegaConf.load(args.config_path)
        transformer_kwargs = self.predict.OmegaConf.to_container(
            self.config["transformer_additional_kwargs"]
        )
        if args.track_head_hidden_dim is not None:
            transformer_kwargs["track_head_hidden_dim"] = int(args.track_head_hidden_dim)
        transformer_kwargs["track_latent_scale"] = float(args.track_latent_scale)

        self.transformer = self.predict.WanTransformer3DModelTrack.from_pretrained(
            os.path.join(
                args.model_name,
                self.config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            ),
            transformer_additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )
        if args.transformer_checkpoint_path:
            resolved_ckpt = self.predict._resolve_transformer_checkpoint(args.transformer_checkpoint_path)
            print(f"[worker] loading finetuned transformer checkpoint: {resolved_ckpt}")
            state_dict = self.predict._load_state_dict_from_path(resolved_ckpt)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for prefix in ("module.", "_orig_mod.", "transformer3d_track.", "transformer."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                cleaned_state_dict[new_key] = value
            missing, unexpected = self.transformer.load_state_dict(cleaned_state_dict, strict=False)
            print(
                f"[worker] checkpoint load done: missing={len(missing)}, unexpected={len(unexpected)}"
            )

        self.vae = self.predict.AutoencoderKLWan.from_pretrained(
            os.path.join(args.model_name, self.config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=self.predict.OmegaConf.to_container(self.config["vae_kwargs"]),
        ).to(self.weight_dtype)
        self.tokenizer = self.predict.AutoTokenizer.from_pretrained(
            os.path.join(
                args.model_name,
                self.config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            ),
        )
        self.text_encoder = self.predict.WanT5EncoderModel.from_pretrained(
            os.path.join(
                args.model_name,
                self.config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            ),
            additional_kwargs=self.predict.OmegaConf.to_container(self.config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).eval()
        self.clip_image_encoder = self.predict.CLIPModel.from_pretrained(
            os.path.join(
                args.model_name,
                self.config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
            ),
        ).to(self.weight_dtype).eval()

        scheduler_cls = {
            "Flow": self.predict.FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": self.predict.FlowUniPCMultistepScheduler,
            "Flow_DPM++": self.predict.FlowDPMSolverMultistepScheduler,
        }[args.sampler_name]
        scheduler_cfg = self.predict.OmegaConf.to_container(self.config["scheduler_kwargs"])
        if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
            scheduler_cfg["shift"] = 1
        scheduler = scheduler_cls(**self.predict.filter_kwargs(scheduler_cls, scheduler_cfg))

        self.pipeline = self.predict.WanFunInpaintPipeline(
            transformer=self.transformer,
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scheduler=scheduler,
            clip_image_encoder=self.clip_image_encoder,
        ).to(device=self.device)
        self.video_length = (
            int(
                (args.video_length - 1)
                // self.vae.config.temporal_compression_ratio
                * self.vae.config.temporal_compression_ratio
            )
            + 1
            if args.video_length != 1
            else 1
        )
        print("[worker] model is resident on GPU; waiting for queued jobs.")

    def _job_args(self, job: JobRecord, job_dir: Path) -> SimpleNamespace:
        negative_text_feature_path = ""
        if job.mode != "cfg" and os.path.isfile(self.args.default_uncond_text_npz):
            negative_text_feature_path = self.args.default_uncond_text_npz
        return SimpleNamespace(
            prompt=job.prompt.strip() or "a video",
            negative_prompt=self.args.negative_prompt,
            validation_image_start=str(job_dir / job.input.image),
            validation_image_end=None,
            track_file_path=str(job_dir / job.input.tracks),
            metadata_path=None,
            sample_index=0,
            random_sample=False,
            train_data_dir=None,
            train_data_root_map_json_track=None,
            train_data_root_id_key_track="root_id",
            use_prompt_from_metadata=False,
            track_condition_index_offset=0,
            text_feature_path="",
            negative_text_feature_path=negative_text_feature_path,
            clip_feature_path="",
            zero_clip_context=bool(self.args.zero_clip_context),
            sample_height=int(self.args.sample_height),
            sample_width=int(self.args.sample_width),
            video_length=int(self.args.video_length),
            fps=int(self.args.fps),
            guidance_mode=job.mode,
            guidance_scale=float(self.args.guidance_scale),
            text_guidance_weight=float(job.text_guidance_weight),
            motion_guidance_weight=float(job.motion_guidance_weight),
            num_inference_steps=int(self.args.num_inference_steps),
            seed=int(job.seed),
            sampler_name=self.args.sampler_name,
            shift=float(self.args.shift),
            mixed_precision=self.args.mixed_precision,
            track_head_hidden_dim=self.args.track_head_hidden_dim,
            save_dir=str(job_dir / "outputs"),
            output_name_suffix=f"{job.mode}_seed{job.seed}_{job.job_id}",
            normalize_track=bool(self.args.track_normalize),
            track_normalize_height=int(self.args.track_normalize_height),
            track_normalize_width=int(self.args.track_normalize_width),
            track_max_points=int(self.args.track_max_points),
            track_point_sample_mode=self.args.track_point_sample_mode,
            track_sort_selected_indices=bool(self.args.track_sort_selected_indices),
            track_point_sample_seed=int(job.seed),
            track_point_id_mode=self.args.track_point_id_mode,
            overlay_linewidth=int(self.args.overlay_linewidth),
            overlay_trace_frames=int(self.args.overlay_trace_frames),
            cotracker_root=self.args.cotracker_root,
            overlay_pad_value=int(self.args.overlay_pad_value),
            debug_track_condition=bool(self.args.debug_track_condition),
            track_analysis=bool(self.args.track_analysis),
            pdb_track_condition=False,
            pdb_pipeline_step0=False,
            force_track_condition_none=False,
            random_fake_track=False,
            track_latent_scale=float(self.args.track_latent_scale),
        )

    def generate(self, job: JobRecord, job_dir: Path) -> None:
        args = self._job_args(job, job_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        effective_track_sample_seed = args.seed
        generator = self.torch.Generator(device=self.device).manual_seed(args.seed)

        input_video, input_video_mask, clip_image = self.predict.get_image_to_video_latent(
            args.validation_image_start,
            args.validation_image_end,
            video_length=self.video_length,
            sample_size=[args.sample_height, args.sample_width],
        )
        if args.zero_clip_context:
            clip_image = None

        track_condition = self.predict._load_track_condition(
            track_file_path=args.track_file_path,
            normalize=args.normalize_track,
            normalize_height=args.track_normalize_height,
            normalize_width=args.track_normalize_width,
            track_max_points=args.track_max_points,
            track_point_sample_mode=args.track_point_sample_mode,
            track_sort_selected_indices=args.track_sort_selected_indices,
            track_point_sample_seed=effective_track_sample_seed,
            track_point_id_mode=args.track_point_id_mode,
            device=self.device,
        )
        if args.debug_track_condition:
            self.predict._debug_log_track_dict(
                "persistent worker: after load (before pipeline)",
                track_condition,
            )

        prompt_embeds = None
        negative_prompt_embeds = None
        if args.text_feature_path:
            prompt_embeds = self.predict._load_text_feature_npz(
                text_feature_path=args.text_feature_path,
                device=self.device,
                dtype=self.weight_dtype,
            )
        if args.negative_text_feature_path:
            negative_prompt_embeds = self.predict._load_text_feature_npz(
                text_feature_path=args.negative_text_feature_path,
                device=self.device,
                dtype=self.weight_dtype,
            )
        elif prompt_embeds is not None:
            negative_prompt_embeds = self.torch.zeros_like(prompt_embeds)

        clip_feature = None
        if args.clip_feature_path and not args.zero_clip_context:
            clip_feature = self.predict._load_clip_feature_npz(
                clip_feature_path=args.clip_feature_path,
                device=self.device,
                dtype=self.weight_dtype,
            )

        if args.debug_track_condition:
            os.environ["WAN_DEBUG_TRACK_CONDITION"] = "1"
        if args.track_analysis:
            os.environ["WAN_TRACK_ANALYSIS"] = "1"
        os.environ["WAN_TRACK_LATENT_SCALE"] = str(args.track_latent_scale)
        try:
            with self.torch.no_grad():
                sample = self.pipeline(
                    prompt=None if prompt_embeds is not None else args.prompt,
                    negative_prompt=None if negative_prompt_embeds is not None else args.negative_prompt,
                    num_frames=self.video_length,
                    height=args.sample_height,
                    width=args.sample_width,
                    generator=generator,
                    guidance_mode=args.guidance_mode,
                    guidance_scale=args.guidance_scale,
                    text_guidance_weight=args.text_guidance_weight,
                    motion_guidance_weight=args.motion_guidance_weight,
                    num_inference_steps=args.num_inference_steps,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    clip_feature=clip_feature,
                    shift=args.shift,
                    track_condition=track_condition,
                ).videos
        finally:
            os.environ.pop("WAN_DEBUG_TRACK_CONDITION", None)
            os.environ.pop("WAN_TRACK_ANALYSIS", None)
            os.environ.pop("WAN_TRACK_LATENT_SCALE", None)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.predict._append_track_analysis_summary(
            save_dir=args.save_dir,
            timestamp=timestamp,
            args=args,
            selected_metadata=None,
            resolved_track_file_path=args.track_file_path,
            transformer=self.transformer,
        )
        suffix = str(args.output_name_suffix).strip()
        suffix_part = f"_{suffix}" if suffix else ""
        output_plain = os.path.join(args.save_dir, f"track_i2v{suffix_part}_{timestamp}.mp4")
        self.predict.save_videos_grid(sample, output_plain, fps=args.fps)
        print(f"[done] saved plain video: {output_plain}")

        raw_tracks, raw_visibility = self.predict._load_track_arrays_raw(
            args.track_file_path,
            track_max_points=args.track_max_points,
            track_point_sample_mode=args.track_point_sample_mode,
            track_sort_selected_indices=args.track_sort_selected_indices,
            track_point_sample_seed=effective_track_sample_seed,
        )
        sample_overlay = self.predict._overlay_tracks_on_video(
            sample=sample,
            tracks=raw_tracks,
            visibility=raw_visibility,
            normalize_track=args.normalize_track,
            normalize_height=args.track_normalize_height,
            normalize_width=args.track_normalize_width,
            overlay_linewidth=args.overlay_linewidth,
            overlay_trace_frames=args.overlay_trace_frames,
            cotracker_root=args.cotracker_root,
            overlay_pad_value=args.overlay_pad_value,
        )
        output_overlay = os.path.join(
            args.save_dir,
            f"track_i2v{suffix_part}_{timestamp}_overlay.mp4",
        )
        self.predict.save_videos_grid(sample_overlay, output_overlay, fps=args.fps)
        print(f"[done] saved track overlay video: {output_overlay}")
        self.torch.cuda.empty_cache()


def run_job(store: JobStore, runtime: TrackI2VPersistentRuntime, job: JobRecord) -> None:
    job_dir = store.job_dir(job.job_id)
    log_path = job_dir / job.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (job_dir / "outputs").mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print(f"[persistent_worker] job_id={job.job_id}")
                print(f"[persistent_worker] mode={job.mode}")
                print("[persistent_worker] model is already resident on GPU")
                print(f"[persistent_worker] save_dir={job_dir / 'outputs'}")
                runtime.generate(job, job_dir)
        latest = store.read_job(job.job_id)
        if latest.status == "canceled":
            store.mark_canceled(job.job_id)
        else:
            store.mark_done(job.job_id, 0)
    except Exception as error:  # noqa: BLE001
        latest = store.read_job(job.job_id)
        if latest.status == "canceled":
            store.mark_canceled(job.job_id)
        else:
            store.mark_failed(job.job_id, None, str(error))


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    jobs_root = Path(args.jobs_root)
    store = JobStore(jobs_root)
    store.recover_after_restart()
    print(f"[worker] repo_root={REPO_ROOT}")
    print(f"[worker] jobs_root={jobs_root}")
    print(f"[worker] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[worker] checkpoint={args.transformer_checkpoint_path}")

    runtime = TrackI2VPersistentRuntime(args)
    while True:
        job = store.claim_next_queued_job()
        if job is None:
            time.sleep(float(args.poll_interval))
            continue
        print(f"[worker] claimed {job.job_id}")
        run_job(store, runtime, job)


if __name__ == "__main__":
    main()
