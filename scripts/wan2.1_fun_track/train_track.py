import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from PIL import Image, ImageDraw

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler

from videox_fun.data import ImageVideoSampler
from videox_fun.data.dataset_image_video_track import (DummyTrackLatentDataset,
                                                       ImageVideoDatasetTrack,
                                                       ImageVideoLatentTrackDataset)
from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.models.wan_transformer3d_track import WanTransformer3DModelTrack
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import filter_kwargs

logger = get_logger(__name__)


def _str2bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _is_new_track_layer_param(name: str, track_condition_mode: str = "track_head") -> bool:
    if track_condition_mode != "track_head":
        return False
    return ("track_head" in name) or name.startswith("patch_embedding.")


def _extract_block_index(name: str) -> Optional[int]:
    m = re.match(r"^blocks\.(\d+)\.", name)
    if m is None:
        return None
    return int(m.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.1 I2V track finetuning scaffold")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_data_meta_track", type=str, default=None)
    parser.add_argument(
        "--val_data_meta_track",
        type=str,
        default=None,
        help="Optional validation metadata path. When set, validation loss can be logged.",
    )
    parser.add_argument(
        "--val_data_dir_track",
        type=str,
        default=None,
        help="Optional validation data root. Defaults to --train_data_dir when omitted.",
    )
    parser.add_argument(
        "--train_data_root_map_json_track",
        type=str,
        default=None,
        help="Optional JSON file with root_id->absolute_root map for multi-root metadata.",
    )
    parser.add_argument(
        "--train_data_root_map_entry_track",
        action="append",
        default=None,
        help="Optional repeated entry in the form root_id=/abs/path (overrides JSON keys).",
    )
    parser.add_argument(
        "--train_data_root_id_key_track",
        type=str,
        default="root_id",
        help="Metadata key name for root id lookup.",
    )
    parser.add_argument("--output_dir_track", type=str, default="output_dir_wan2.1_fun_track")
    parser.add_argument(
        "--checkpoint_dir_track",
        type=str,
        default=None,
        help=(
            "Directory for accelerator checkpoints (checkpoint-N folders). "
            "Defaults to --output_dir_track when omitted."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint_track",
        type=str,
        default=None,
        help=(
            "Resume training: path to a checkpoint folder (e.g. .../checkpoint-1750), or "
            '"latest" to pick the highest checkpoint-N under --checkpoint_dir_track '
            "(or --output_dir_track if no separate checkpoint dir)."
        ),
    )
    parser.add_argument(
        "--init_model_from_checkpoint_track",
        type=str,
        default=None,
        help=(
            "Initialize only model weights from an accelerate checkpoint folder "
            "(e.g. .../checkpoint-600), or \"latest\" under --checkpoint_dir_track "
            "(or --output_dir_track). Optimizer/lr scheduler/global_step are reset."
        ),
    )
    parser.add_argument("--input_mode_track", type=str, default="video", choices=["video", "latent"])
    parser.add_argument("--latent_file_key_track", type=str, default="latent_file_path")
    parser.add_argument(
        "--first_frame_vae_latent_file_key_track",
        type=str,
        default="first_frame_vae_latent_file_path",
        help=(
            "Optional metadata key for precomputed first-frame VAE latent tensor "
            "([C,F,H,W] or [1,C,F,H,W])."
        ),
    )
    parser.add_argument(
        "--use_first_frame_condition_track",
        action="store_true",
        help="In latent mode, build inpaint condition y from first_frame.png via VAE encode.",
    )
    parser.add_argument(
        "--verify_first_frame_vae_latent_track",
        action="store_true",
        help=(
            "When precomputed first_frame_vae_latent is available, also build online first-frame "
            "latent and compare differences for sanity checks."
        ),
    )
    parser.add_argument(
        "--verify_first_frame_vae_latent_max_batches",
        type=int,
        default=0,
        help="Max number of batches to verify in one run. <=0 means 1 batch when verify is enabled.",
    )
    parser.add_argument(
        "--verify_first_frame_vae_latent_tol",
        type=float,
        default=1e-3,
        help="Absolute max-diff tolerance threshold for verification warning.",
    )

    parser.add_argument("--train_mode", type=str, default="inpaint", choices=["normal", "inpaint", "i2v"])
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--video_sample_stride", type=int, default=2)
    parser.add_argument("--video_sample_n_frames", type=int, default=81)
    parser.add_argument("--video_sample_size", type=int, default=640)
    parser.add_argument("--image_sample_size", type=int, default=640)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument(
        "--validation_steps_track",
        type=int,
        default=0,
        help="Run validation every N optimizer steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--validation_max_batches_track",
        type=int,
        default=8,
        help="Max validation batches per validation run.",
    )
    parser.add_argument(
        "--validation_subset_size_track",
        type=int,
        default=0,
        help=(
            "If >0, evaluate a fixed random subset of this many validation samples "
            "instead of the metadata prefix. The subset is selected once at startup."
        ),
    )
    parser.add_argument(
        "--validation_subset_seed_track",
        type=int,
        default=42,
        help="Seed used to select --validation_subset_size_track samples.",
    )
    parser.add_argument(
        "--validation_seed_track",
        type=int,
        default=42,
        help="Seed used for validation noise/timestep sampling; fixed across validation runs.",
    )
    parser.add_argument(
        "--validation_track_max_points_track",
        type=int,
        default=-1,
        help=(
            "Deterministic max track points for validation. >0 keeps the first N points; "
            "<=0 keeps all points allowed by --track_max_points."
        ),
    )

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--new_track_layers_lr",
        type=float,
        default=None,
        help=(
            "Optional LR for newly-added track layers (track_head + patch_embedding.*). "
            "Defaults to --learning_rate when omitted."
        ),
    )
    parser.add_argument(
        "--early_blocks_lr",
        type=float,
        default=None,
        help=(
            "Optional LR for selected pretrained early transformer blocks. "
            "Defaults to --learning_rate when omitted."
        ),
    )
    parser.add_argument(
        "--train_early_blocks_track",
        type=int,
        default=-1,
        help=(
            "If >=0, train only blocks.[0..N-1] among pretrained blocks, while always allowing "
            "newly-added track layers included by --trainable_modules_track."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=3e-2,
        help="AdamW weight decay coefficient.",
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--gradient_checkpointing",
        type=_str2bool,
        default=False,
        help="Enable block-level gradient checkpointing for transformer blocks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'Where to log metrics. Use "tensorboard" (default), "wandb", "comet_ml", or "all" '
            "(requires the corresponding package). For Weights & Biases, install `wandb` and set "
            "`WANDB_API_KEY` (or run `wandb login`)."
        ),
    )
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name_track", type=str, default="wan2.1_fun_track")
    parser.add_argument(
        "--wandb_run_name_track",
        type=str,
        default=None,
        help='Optional Weights & Biases run name (only used when --report_to is "wandb" or "all").',
    )

    parser.add_argument("--trainable_modules_track", nargs="+", default=["."])
    parser.add_argument(
        "--new_params_only_steps_track",
        type=int,
        default=0,
        help=(
            "If >0, train only newly initialized track params for the first N optimizer steps, "
            "then unfreeze all parameters matched by --trainable_modules_track."
        ),
    )
    parser.add_argument("--tokenizer_max_length", type=int, default=512)
    parser.add_argument(
        "--text_drop_ratio_track",
        type=float,
        default=0.1,
        help=(
            "Per-sample caption dropout for train (probability of empty text). "
            "Latent+precomputed: collate swaps dropped rows for null T5 embeds when enabled. "
            "Video mode: forwarded to ImageVideoDatasetTrack. Validation always uses 0."
        ),
    )
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")

    parser.add_argument("--use_track_condition", action="store_true")
    parser.add_argument(
        "--track_condition_mode",
        type=str,
        default="track_head",
        choices=["track_head", "wan_move"],
        help=(
            "track_head keeps the existing learned track_head/extra-channel concat path. "
            "wan_move keeps the base Wan transformer and injects motion by copying first-frame "
            "VAE features along tracks inside the inpaint y condition."
        ),
    )
    parser.add_argument(
        "--apply_track_patch_embed_init_track",
        type=_str2bool,
        default=True,
        help=(
            "When --use_track_condition is on, apply post-load scaled duplicate init to the "
            "newly added track patch-embed channels."
        ),
    )
    parser.add_argument(
        "--track_patch_init_alpha",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --track_patch_init_gain. When provided, it is used only if "
            "--track_patch_init_gain is omitted."
        ),
    )
    parser.add_argument(
        "--track_patch_init_mode",
        type=str,
        default="copy_noisy",
        choices=["copy_noisy", "copy_first", "avg_noisy_first"],
        help=(
            "Initialization source for the added track patch-embed slice: copy_noisy copies "
            "old weight[:,0:16], copy_first copies old weight[:,20:36], avg_noisy_first "
            "copies their 0.5/0.5 average."
        ),
    )
    parser.add_argument(
        "--track_patch_init_gain",
        type=float,
        default=None,
        help="Optional gain applied only to the added track patch-embed slice. Defaults to 1.0.",
    )
    parser.add_argument(
        "--track_latent_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to track_head output before concat. This is part of model config, "
            "so train and inference can use the same value."
        ),
    )
    parser.add_argument(
        "--add_track_init_noise",
        type=_str2bool,
        default=False,
        help="Add tiny noise to the track patch-init block for symmetry breaking.",
    )
    parser.add_argument(
        "--track_init_noise_scale",
        type=float,
        default=0.01,
        help="Noise scale used when --add_track_init_noise=true.",
    )
    parser.add_argument(
        "--track_head_hidden_dim",
        type=int,
        default=None,
        help=(
            "Override track head hidden dim (Conv3D bottleneck width). "
            "Defaults to checkpoint/config value when omitted."
        ),
    )
    parser.add_argument("--track_condition_key", type=str, default="track_condition")
    parser.add_argument("--track_max_points", type=int, default=-1)
    parser.add_argument(
        "--track_random_points_min",
        type=int,
        default=0,
        help="If >0 with --track_random_points_max, randomly sample points per batch from [min,max].",
    )
    parser.add_argument(
        "--track_random_points_max",
        type=int,
        default=0,
        help="If >0 with --track_random_points_min, randomly sample points per batch from [min,max].",
    )
    parser.add_argument(
        "--track_sort_selected_indices",
        type=_str2bool,
        default=True,
        help=(
            "When random track-point sampling is enabled, sort sampled point indices "
            "to preserve original index order. Set false to keep random permutation order."
        ),
    )
    parser.add_argument(
        "--track_point_id_mode",
        type=str,
        default="original",
        choices=["original", "local"],
        help=(
            "How to assign point_ids for track positional/identity embeddings. "
            "original keeps source point indices after sampling; local reindexes the "
            "sampled subset to 0..P-1."
        ),
    )
    parser.add_argument("--track_normalize", action="store_true")
    parser.add_argument("--track_normalize_height", type=int, default=480)
    parser.add_argument("--track_normalize_width", type=int, default=832)
    parser.add_argument(
        "--track_condition_drop_prob",
        type=float,
        default=0.0,
        help=(
            "Per-sample probability of dropping track conditioning during training. "
            "Dropped samples keep tensor shapes but receive zeroed visibility/point_mask, "
            "so the resulting track canvas becomes zero padding."
        ),
    )
    parser.add_argument(
        "--first_frame_condition_drop_prob_track",
        type=float,
        default=0.0,
        help=(
            "Per-sample probability of dropping first-frame conditioning during training. "
            "Dropped rows zero-out both inpaint condition y and precomputed clip_feature "
            "for latent-mode train batches."
        ),
    )
    parser.add_argument(
        "--no_apply_text_dropout_to_precomputed_track",
        action="store_true",
        help=(
            "Latent mode only: by default, when caption dropout clears text to empty, "
            "precomputed T5 rows are replaced with null embeddings (from "
            "--precomputed_uncond_text_npz_track if set, else encoded '' at startup). "
            "Pass this flag to keep the on-disk caption precomputes (old behavior)."
        ),
    )
    parser.add_argument(
        "--precomputed_uncond_text_npz_track",
        type=str,
        default=None,
        help=(
            "Latent mode: npz with prompt_embeds [L,D] and optional attention_mask [L] "
            "(same keys as text_feature_wan_t5.npz). When set, used for dropped-caption rows "
            "instead of running T5 on '' at startup. See "
            "scripts/wan2.1_fun_track/precompute_uncond_text_track.py."
        ),
    )

    parser.add_argument("--dry_run_track", action="store_true")
    parser.add_argument("--dummy_data_track", action="store_true")
    parser.add_argument("--dummy_length_track", type=int, default=128)
    parser.add_argument("--dummy_n_frames_track", type=int, default=81)
    parser.add_argument("--dummy_n_points_track", type=int, default=50)
    parser.add_argument("--dummy_latent_channels_track", type=int, default=16)
    parser.add_argument("--dummy_latent_frames_track", type=int, default=21)
    parser.add_argument("--dummy_latent_h_track", type=int, default=60)
    parser.add_argument("--dummy_latent_w_track", type=int, default=104)
    parser.add_argument(
        "--debug_memory_track",
        action="store_true",
        help="Log trainable parameter counts and per-step CUDA memory stats.",
    )
    parser.add_argument(
        "--debug_weight_update_track",
        action="store_true",
        help="Log which trainable parameters actually receive gradients and updates.",
    )
    parser.add_argument(
        "--debug_weight_update_topk_track",
        type=int,
        default=30,
        help="Top-K parameters to print by update norm when --debug_weight_update_track is enabled.",
    )
    parser.add_argument(
        "--track_debug_vis_steps",
        type=int,
        default=0,
        help=(
            "If >0, save track-debug visualizations every N optimizer steps "
            "(pixel-track, latent-canvas, decoded-latent-overlay)."
        ),
    )
    parser.add_argument(
        "--track_debug_vis_dir",
        type=str,
        default="track_debug_vis",
        help="Subdirectory under --output_dir_track for debug visualization artifacts.",
    )
    parser.add_argument(
        "--track_debug_vis_sample_index",
        type=int,
        default=0,
        help="Batch sample index used for debug visualization export.",
    )
    parser.add_argument(
        "--track_debug_vis_max_frames",
        type=int,
        default=32,
        help="Maximum number of frames to export per debug visualization clip.",
    )
    parser.add_argument(
        "--track_debug_vis_max_points",
        type=int,
        default=512,
        help="Maximum visible track points to draw per frame. <=0 means all valid points.",
    )
    parser.add_argument(
        "--track_debug_vis_fps",
        type=int,
        default=8,
        help="FPS for saved debug visualization MP4 clips.",
    )
    return parser.parse_args()


def _parse_root_map(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    root_map: Dict[str, str] = {}
    if args.train_data_root_map_json_track is not None:
        with open(args.train_data_root_map_json_track, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--train_data_root_map_json_track must point to a JSON object.")
        for key, value in loaded.items():
            key_s = str(key).strip()
            value_s = str(value).strip()
            if key_s == "" or value_s == "":
                continue
            root_map[key_s] = value_s

    if args.train_data_root_map_entry_track is not None:
        for entry in args.train_data_root_map_entry_track:
            if "=" not in entry:
                raise ValueError(
                    f"Invalid --train_data_root_map_entry_track value '{entry}'. "
                    "Expected root_id=/abs/path"
                )
            key, value = entry.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "" or value == "":
                raise ValueError(
                    f"Invalid --train_data_root_map_entry_track value '{entry}'. "
                    "Both root_id and path must be non-empty."
                )
            root_map[key] = value

    if len(root_map) == 0:
        return None
    return root_map


def _sniff_metadata_source_media(meta_path: str, max_bytes: int = 1_000_000) -> Optional[str]:
    """Best-effort metadata mode sniffing without full JSON parsing."""
    try:
        with open(meta_path, "rb") as f:
            chunk = f.read(max_bytes)
    except OSError:
        return None

    text = chunk.decode("utf-8", errors="ignore").lower()
    compact = "".join(text.split())

    if '"source_media":"latent"' in compact:
        return "latent"
    if '"source_media":"video"' in compact:
        return "video"

    # Fallback heuristics for older metadata variants.
    if '"latent_file_path"' in compact or "vae_latents.pt" in text:
        return "latent"
    if ".mp4" in text:
        return "video"
    return None


def _make_fixed_subset_indices(length: int, subset_size: int, seed: int) -> List[int]:
    subset_size = min(max(0, int(subset_size)), int(length))
    if subset_size <= 0:
        return []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(int(length), generator=generator)[:subset_size].tolist()


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _pad_track_condition(
    track_batch: List[Optional[Dict[str, torch.Tensor]]],
    max_points: int,
    random_points_min: int,
    random_points_max: int,
    sort_selected_indices: bool,
    point_id_mode: str,
    normalize: bool,
    h: int,
    w: int,
    random_points: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    valid_items = [item for item in track_batch if item is not None]
    if not valid_items:
        return None

    device = valid_items[0]["tracks"].device
    dtype = valid_items[0]["tracks"].dtype
    bsz = len(track_batch)
    t = max(item["tracks"].shape[0] for item in valid_items)
    max_points_in_batch = max(item["tracks"].shape[1] for item in valid_items)
    hard_cap = max_points_in_batch if int(max_points) <= 0 else min(max_points_in_batch, int(max_points))

    rp_min = int(random_points_min)
    rp_max = int(random_points_max)
    if random_points and rp_min > 0 and rp_max > 0:
        lo = min(rp_min, rp_max)
        hi = max(rp_min, rp_max)
        lo = min(lo, hard_cap)
        hi = min(hi, hard_cap)
        if lo <= 0:
            lo = 1
        if hi >= lo:
            p = int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())
        else:
            p = hard_cap
    else:
        p = hard_cap

    point_id_mode = str(point_id_mode).strip().lower()
    if point_id_mode not in {"original", "local"}:
        raise ValueError(f"Unsupported point_id_mode: {point_id_mode}")

    tracks = torch.zeros((bsz, t, p, 2), dtype=dtype, device=device)
    visibility = torch.zeros((bsz, t, p), dtype=dtype, device=device)
    point_mask = torch.zeros((bsz, p), dtype=torch.bool, device=device)
    point_ids = torch.full((bsz, p), -1, dtype=torch.long, device=device)

    for i, item in enumerate(track_batch):
        if item is None:
            continue
        cur_tracks_all = item["tracks"][:t]
        cur_vis_all = item["visibility"][:t]
        cur_points = int(cur_tracks_all.shape[1])
        keep = min(p, cur_points)
        if cur_points > keep:
            if random_points:
                selected = torch.randperm(cur_points, device=cur_tracks_all.device)[:keep]
                if sort_selected_indices:
                    selected, _ = torch.sort(selected)
            else:
                selected = torch.arange(keep, device=cur_tracks_all.device, dtype=torch.long)
            cur_tracks = cur_tracks_all[:, selected]
            cur_vis = cur_vis_all[:, selected]
            if point_id_mode == "local":
                cur_point_ids = torch.arange(keep, device=cur_tracks_all.device, dtype=torch.long)
            else:
                cur_point_ids = selected.to(dtype=torch.long)
        else:
            cur_tracks = cur_tracks_all[:, :keep]
            cur_vis = cur_vis_all[:, :keep]
            cur_point_ids = torch.arange(keep, device=cur_tracks_all.device, dtype=torch.long)

        tracks[i, :cur_tracks.shape[0], :cur_tracks.shape[1]] = cur_tracks
        visibility[i, :cur_vis.shape[0], :cur_vis.shape[1]] = cur_vis
        point_mask[i, :cur_tracks.shape[1]] = True
        point_ids[i, :cur_tracks.shape[1]] = cur_point_ids

    if normalize:
        tracks[..., 0] = tracks[..., 0] / float(max(w, 1))
        tracks[..., 1] = tracks[..., 1] / float(max(h, 1))

    return {
        "tracks": tracks,
        "visibility": visibility,
        "point_mask": point_mask,
        "point_ids": point_ids,
        "is_normalized": torch.full(
            (bsz,),
            bool(normalize),
            dtype=torch.bool,
            device=device,
        ),
        "track_resolution": torch.tensor(
            [[float(max(w, 1)), float(max(h, 1))]] * bsz,
            dtype=torch.float32,
            device=device,
        ),
    }


def _apply_track_condition_dropout(
    track_condition: Optional[Dict[str, torch.Tensor]],
    drop_prob: float,
) -> Optional[Dict[str, torch.Tensor]]:
    if track_condition is None:
        return None
    p = float(drop_prob)
    if p <= 0.0:
        return track_condition
    if p >= 1.0:
        drop_mask = torch.ones(
            (track_condition["visibility"].shape[0],),
            dtype=torch.bool,
            device=track_condition["visibility"].device,
        )
    else:
        drop_mask = torch.rand(
            (track_condition["visibility"].shape[0],),
            device=track_condition["visibility"].device,
        ) < p
    if not torch.any(drop_mask):
        return track_condition

    dropped = dict(track_condition)
    dropped["visibility"] = track_condition["visibility"].clone()
    dropped["visibility"][drop_mask] = 0

    point_mask = track_condition.get("point_mask", None)
    if point_mask is not None:
        dropped["point_mask"] = point_mask.clone()
        dropped["point_mask"][drop_mask] = False
    return dropped


def _empty_wan_move_condition_stats(enabled: bool = False) -> Dict[str, float]:
    return {
        "wan_move_condition/enabled": float(enabled),
        "wan_move_condition/copied_sites": 0.0,
        "wan_move_condition/valid_source_points": 0.0,
        "wan_move_condition/valid_target_sites": 0.0,
        "wan_move_condition/copied_site_ratio": 0.0,
    }


def _as_batched_bool_hint(
    value,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    hint = value.to(device=device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    if hint.ndim == 0:
        hint = hint.view(1)
    if hint.ndim != 1:
        raise ValueError(f"is_normalized must be scalar or [B], got {tuple(hint.shape)}")
    if hint.shape[0] == 1 and batch_size > 1:
        hint = hint.expand(batch_size)
    if hint.shape[0] != batch_size:
        raise ValueError(f"is_normalized batch {hint.shape[0]} != tracks batch {batch_size}")
    return hint.to(dtype=torch.bool)


def _as_batched_track_resolution(
    value,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    resolution = value.to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=torch.float32)
    if resolution.ndim == 1:
        resolution = resolution.unsqueeze(0)
    if resolution.ndim != 2 or resolution.shape[-1] < 2:
        raise ValueError(f"track_resolution must be [B,2] or [2], got {tuple(resolution.shape)}")
    resolution = resolution[:, :2]
    if resolution.shape[0] == 1 and batch_size > 1:
        resolution = resolution.expand(batch_size, -1)
    if resolution.shape[0] != batch_size:
        raise ValueError(f"track_resolution batch {resolution.shape[0]} != tracks batch {batch_size}")
    return resolution


def _map_wan_move_tracks_to_latent_grid(
    tracks: torch.Tensor,
    h_lat: int,
    w_lat: int,
    is_normalized_hint,
    track_resolution,
) -> torch.Tensor:
    bsz = int(tracks.shape[0])
    device = tracks.device
    x = tracks[..., 0]
    y = tracks[..., 1]

    hint = _as_batched_bool_hint(is_normalized_hint, bsz, device=device)
    if hint is None:
        is_normalized_batch = torch.full(
            (bsz,),
            bool((tracks.max() <= 2.0).item() and (tracks.min() >= -0.5).item()),
            device=device,
            dtype=torch.bool,
        )
    else:
        is_normalized_batch = hint

    resolution = _as_batched_track_resolution(track_resolution, bsz, device=device)
    if resolution is None:
        # Fallback for legacy pixel tracks without source resolution metadata.
        src_w = torch.clamp(x.amax(dim=(1, 2)).view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(y.amax(dim=(1, 2)).view(-1, 1, 1), min=1.0)
    else:
        src_w = torch.clamp(resolution[:, 0].view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(resolution[:, 1].view(-1, 1, 1), min=1.0)

    # Wan-Move first scales pixel tracks to the target frame size, then uses x // 8, y // 8.
    # With normalized tracks this is equivalent to floor(x_norm * W_lat).
    gx_norm = torch.floor(x * float(max(w_lat, 1)))
    gy_norm = torch.floor(y * float(max(h_lat, 1)))
    gx_pixel = torch.floor(x / src_w * float(max(w_lat, 1)))
    gy_pixel = torch.floor(y / src_h * float(max(h_lat, 1)))

    norm_mask = is_normalized_batch.view(-1, 1, 1)
    gx = torch.where(norm_mask, gx_norm, gx_pixel)
    gy = torch.where(norm_mask, gy_norm, gy_pixel)
    return torch.stack([gx, gy], dim=-1).long()


def _apply_wan_move_feature_replace(
    condition_latents: torch.Tensor,
    track_condition: Optional[Dict[str, torch.Tensor]],
    temporal_stride: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    stats = _empty_wan_move_condition_stats(enabled=True)
    if track_condition is None:
        return condition_latents, stats

    tracks = track_condition.get("tracks", None)
    visibility = track_condition.get("visibility", None)
    if tracks is None or visibility is None:
        return condition_latents, stats

    device = condition_latents.device
    bsz, _, latent_frames, h_lat, w_lat = condition_latents.shape
    tracks = tracks.to(device=device, dtype=torch.float32)
    visibility = visibility.to(device=device, dtype=torch.float32)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be [B,T,P,2], got {tuple(tracks.shape)}")
    if visibility.ndim != 3:
        raise ValueError(f"visibility must be [B,T,P], got {tuple(visibility.shape)}")
    if tracks.shape[0] != bsz:
        raise ValueError(f"Batch mismatch between condition latents ({bsz}) and tracks ({tracks.shape[0]}).")
    if visibility.shape[:3] != tracks.shape[:3]:
        raise ValueError(
            "tracks/visibility shape mismatch: "
            f"tracks={tuple(tracks.shape)} visibility={tuple(visibility.shape)}"
        )

    point_mask = track_condition.get("point_mask", None)
    if point_mask is None:
        point_mask = torch.ones((bsz, tracks.shape[2]), device=device, dtype=torch.bool)
    else:
        point_mask = point_mask.to(device=device, dtype=torch.bool)
        if point_mask.shape != (bsz, tracks.shape[2]):
            raise ValueError(f"point_mask must be [B,P], got {tuple(point_mask.shape)}")

    stride = max(1, int(temporal_stride))
    frame_idx = torch.arange(latent_frames, device=device, dtype=torch.long) * stride
    frame_idx = torch.clamp(frame_idx, max=max(int(tracks.shape[1]) - 1, 0))

    sampled_tracks = tracks.index_select(1, frame_idx)
    sampled_visibility = visibility.index_select(1, frame_idx) > 0.5
    grid_xy = _map_wan_move_tracks_to_latent_grid(
        sampled_tracks,
        h_lat=h_lat,
        w_lat=w_lat,
        is_normalized_hint=track_condition.get("is_normalized", None),
        track_resolution=track_condition.get("track_resolution", None),
    )
    gx = grid_xy[..., 0]
    gy = grid_xy[..., 1]
    in_bound = (gx >= 0) & (gx < w_lat) & (gy >= 0) & (gy < h_lat)
    valid = sampled_visibility & in_bound & point_mask[:, None, :]
    if latent_frames <= 1:
        return condition_latents, stats

    source_valid = valid[:, 0, :]
    target_valid = valid[:, 1:, :] & source_valid[:, None, :]
    valid_indices = target_valid.nonzero(as_tuple=False)
    valid_source_points = int(source_valid.sum().item())
    valid_target_sites = int(target_valid.sum().item())
    stats["wan_move_condition/valid_source_points"] = float(valid_source_points)
    stats["wan_move_condition/valid_target_sites"] = float(valid_target_sites)
    if valid_indices.numel() == 0:
        return condition_latents, stats

    edited = condition_latents.clone()
    batch_idx = valid_indices[:, 0]
    t_rel = valid_indices[:, 1]
    point_idx = valid_indices[:, 2]
    t_target = t_rel + 1

    h_target = gy[batch_idx, t_target, point_idx].long()
    w_target = gx[batch_idx, t_target, point_idx].long()
    h_source = gy[batch_idx, 0, point_idx].long()
    w_source = gx[batch_idx, 0, point_idx].long()

    src_features = edited[batch_idx, :, 0, h_source, w_source]
    edited[batch_idx, :, t_target, h_target, w_target] = src_features

    copied = int(valid_indices.shape[0])
    stats["wan_move_condition/copied_sites"] = float(copied)
    stats["wan_move_condition/copied_site_ratio"] = float(copied) / float(max(valid_target_sites, 1))
    return edited, stats


def _forward_transformer_with_track_mode(
    transformer: torch.nn.Module,
    track_condition_mode: str,
    track_condition: Optional[Dict[str, torch.Tensor]],
    **kwargs,
) -> torch.Tensor:
    if track_condition_mode == "track_head":
        return transformer(**kwargs, track_condition=track_condition)
    return transformer(**kwargs)


def _summarize_track_condition(
    track_condition: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    if track_condition is None:
        return {
            "track_stats/batch_has_track": 0.0,
            "track_stats/point_count_avg": 0.0,
            "track_stats/visible_ratio": 0.0,
            "track_stats/out_of_bounds_ratio": 0.0,
            "track_stats/dropped_sample_ratio": 1.0,
        }

    tracks = track_condition["tracks"].detach().float()
    visibility = track_condition["visibility"].detach().float()
    point_mask = track_condition.get("point_mask", None)
    is_normalized_hint = track_condition.get("is_normalized", None)
    track_resolution = track_condition.get("track_resolution", None)

    if point_mask is None:
        point_mask = torch.ones(
            (tracks.shape[0], tracks.shape[2]),
            dtype=torch.bool,
            device=tracks.device,
        )
    else:
        point_mask = point_mask.detach().to(device=tracks.device, dtype=torch.bool)

    valid_points = point_mask[:, None, :].expand(-1, tracks.shape[1], -1)
    visible_bool = (visibility > 0.5) & valid_points
    valid_point_count = point_mask.sum(dim=1).float()
    dropped_samples = (valid_point_count == 0).float()

    if is_normalized_hint is not None:
        if isinstance(is_normalized_hint, torch.Tensor):
            is_norm = is_normalized_hint.detach().to(device=tracks.device)
        else:
            is_norm = torch.as_tensor(is_normalized_hint, device=tracks.device)
        if is_norm.ndim == 0:
            is_norm = is_norm.view(1)
        if is_norm.ndim != 1:
            raise ValueError(
                f"is_normalized must be scalar or [B], got {tuple(is_norm.shape)}"
            )
        if is_norm.shape[0] == 1 and tracks.shape[0] > 1:
            is_norm = is_norm.expand(tracks.shape[0])
        if is_norm.shape[0] != tracks.shape[0]:
            raise ValueError(
                f"is_normalized batch {is_norm.shape[0]} != tracks batch {tracks.shape[0]}"
            )
        is_norm = is_norm.to(dtype=torch.bool).view(-1, 1, 1)

        x_normed = tracks[..., 0]
        y_normed = tracks[..., 1]
        if track_resolution is not None:
            track_resolution = track_resolution.detach().float().to(device=tracks.device)
            if track_resolution.ndim == 1:
                track_resolution = track_resolution.unsqueeze(0)
            src_w = torch.clamp(track_resolution[:, 0].view(-1, 1, 1), min=1.0)
            src_h = torch.clamp(track_resolution[:, 1].view(-1, 1, 1), min=1.0)
            x_from_pixel = tracks[..., 0] / src_w
            y_from_pixel = tracks[..., 1] / src_h
        else:
            x_from_pixel = tracks[..., 0]
            y_from_pixel = tracks[..., 1]
        x_norm = torch.where(is_norm, x_normed, x_from_pixel)
        y_norm = torch.where(is_norm, y_normed, y_from_pixel)
    elif track_resolution is not None:
        track_resolution = track_resolution.detach().float().to(device=tracks.device)
        if track_resolution.ndim == 1:
            track_resolution = track_resolution.unsqueeze(0)
        src_w = torch.clamp(track_resolution[:, 0].view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(track_resolution[:, 1].view(-1, 1, 1), min=1.0)
        x_norm = tracks[..., 0] / src_w
        y_norm = tracks[..., 1] / src_h
    else:
        x_norm = tracks[..., 0]
        y_norm = tracks[..., 1]

    oob = ((x_norm < 0.0) | (x_norm > 1.0) | (y_norm < 0.0) | (y_norm > 1.0)) & valid_points
    valid_points_count = valid_points.sum().item()

    return {
        "track_stats/batch_has_track": 1.0,
        "track_stats/point_count_avg": float(valid_point_count.mean().item()),
        "track_stats/visible_ratio": float(visible_bool.float().mean().item()),
        "track_stats/out_of_bounds_ratio": float(
            oob.float().sum().item() / max(valid_points_count, 1)
        ),
        "track_stats/dropped_sample_ratio": float(dropped_samples.mean().item()),
    }


def _ensure_even_hw_uint8_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected frame [H,W,3], got {tuple(frame.shape)}")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _save_video_mp4_uint8(frames_uint8: np.ndarray, output_path: str, fps: int) -> None:
    if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
        raise ValueError(
            f"frames_uint8 must be [T,H,W,3], got {tuple(frames_uint8.shape)}"
        )
    if int(frames_uint8.shape[0]) <= 0:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = imageio.get_writer(
        output_path,
        fps=max(1, int(fps)),
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames_uint8:
            writer.append_data(_ensure_even_hw_uint8_frame(frame))
    finally:
        writer.close()


def _video_tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected [1,C,F,H,W] for batched video, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [C,F,H,W], got {tuple(video.shape)}")

    video_f = video.detach().float().cpu().permute(1, 2, 3, 0).contiguous()  # [F,H,W,C]
    vmin = float(video_f.min().item())
    vmax = float(video_f.max().item())
    if vmin < -0.25 or vmax > 1.25:
        video_f = (video_f / 2.0) + 0.5
    video_f = video_f.clamp(0.0, 1.0)
    return (video_f * 255.0).round().to(torch.uint8).numpy()


def _select_track_point_indices_for_vis(
    point_mask: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    valid_indices = torch.nonzero(point_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return valid_indices
    if int(max_points) <= 0 or valid_indices.numel() <= int(max_points):
        return valid_indices
    sample_positions = torch.linspace(
        0,
        valid_indices.numel() - 1,
        steps=int(max_points),
        dtype=torch.float32,
    ).round().long()
    return valid_indices[sample_positions]


def _extract_track_sample_for_vis(
    track_condition: Dict[str, torch.Tensor],
    sample_index: int,
    max_points: int,
    default_w: int,
    default_h: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], int]:
    tracks_b = track_condition["tracks"].detach().float().cpu()
    visibility_b = track_condition["visibility"].detach().float().cpu()
    if tracks_b.ndim != 4 or tracks_b.shape[-1] != 2:
        raise ValueError(f"tracks must be [B,T,P,2], got {tuple(tracks_b.shape)}")
    if visibility_b.ndim != 3:
        raise ValueError(f"visibility must be [B,T,P], got {tuple(visibility_b.shape)}")

    bsz = int(tracks_b.shape[0])
    sample_idx = min(max(int(sample_index), 0), max(bsz - 1, 0))

    point_mask_b = track_condition.get("point_mask", None)
    if point_mask_b is None:
        point_mask = torch.ones((tracks_b.shape[2],), dtype=torch.bool)
    else:
        point_mask_full = torch.as_tensor(point_mask_b).detach().cpu()
        if point_mask_full.ndim == 1:
            point_mask = point_mask_full.to(dtype=torch.bool)
        elif point_mask_full.ndim == 2:
            pm_idx = min(sample_idx, point_mask_full.shape[0] - 1)
            point_mask = point_mask_full[pm_idx].to(dtype=torch.bool)
        else:
            raise ValueError(
                f"point_mask must be [P] or [B,P], got {tuple(point_mask_full.shape)}"
            )

    keep_indices = _select_track_point_indices_for_vis(point_mask, max_points=max_points)
    tracks = tracks_b[sample_idx, :, keep_indices].clone()
    visibility = visibility_b[sample_idx, :, keep_indices].clone()

    src_w = float(max(default_w, 1))
    src_h = float(max(default_h, 1))
    track_resolution = track_condition.get("track_resolution", None)
    if track_resolution is not None:
        tr = torch.as_tensor(track_resolution).detach().float().cpu()
        if tr.ndim == 1 and tr.numel() >= 2:
            src_w = float(max(tr[0].item(), 1.0))
            src_h = float(max(tr[1].item(), 1.0))
        elif tr.ndim == 2 and tr.shape[1] >= 2:
            tr_idx = min(sample_idx, tr.shape[0] - 1)
            src_w = float(max(tr[tr_idx, 0].item(), 1.0))
            src_h = float(max(tr[tr_idx, 1].item(), 1.0))

    is_normalized = False
    is_normalized_hint = track_condition.get("is_normalized", None)
    if is_normalized_hint is not None:
        hint = torch.as_tensor(is_normalized_hint).detach().cpu()
        if hint.ndim == 0:
            is_normalized = bool(hint.item())
        elif hint.ndim >= 1:
            is_normalized = bool(hint.view(-1)[min(sample_idx, hint.view(-1).numel() - 1)].item())

    if is_normalized:
        tracks[..., 0] = tracks[..., 0] * src_w
        tracks[..., 1] = tracks[..., 1] * src_h

    return tracks, visibility, (int(round(src_w)), int(round(src_h))), sample_idx


def _draw_visible_track_points(
    frame_rgb: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    visible_mask: np.ndarray,
    radius: int = 2,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    if frame_rgb.dtype != np.uint8:
        raise ValueError(f"frame_rgb dtype must be uint8, got {frame_rgb.dtype}")
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frame [H,W,3], got {tuple(frame_rgb.shape)}")

    h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    image = Image.fromarray(frame_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    valid_indices = np.nonzero(visible_mask)[0]
    rad = max(int(radius), 1)
    for idx in valid_indices:
        x_val = float(x_coords[idx])
        y_val = float(y_coords[idx])
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        x_i = int(round(x_val))
        y_i = int(round(y_val))
        if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
            continue
        draw.ellipse(
            (x_i - rad, y_i - rad, x_i + rad, y_i + rad),
            fill=color,
        )
    return np.asarray(image, dtype=np.uint8)


def _render_track_points_video(
    tracks_px: torch.Tensor,
    visibility: torch.Tensor,
    canvas_h: int,
    canvas_w: int,
    max_frames: int,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    if tracks_px.ndim != 3 or tracks_px.shape[-1] != 2:
        raise ValueError(f"tracks_px must be [T,P,2], got {tuple(tracks_px.shape)}")
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be [T,P], got {tuple(visibility.shape)}")

    t_total = int(min(tracks_px.shape[0], visibility.shape[0]))
    t_vis = t_total if int(max_frames) <= 0 else min(t_total, int(max_frames))
    frames = np.zeros(
        (t_vis, int(max(canvas_h, 1)), int(max(canvas_w, 1)), 3),
        dtype=np.uint8,
    )
    tracks_np = tracks_px.detach().cpu().numpy()
    visibility_np = visibility.detach().cpu().numpy()

    for frame_idx in range(t_vis):
        x_vals = tracks_np[frame_idx, :, 0]
        y_vals = tracks_np[frame_idx, :, 1]
        vis_mask = visibility_np[frame_idx] > 0.5
        frames[frame_idx] = _draw_visible_track_points(
            frame_rgb=frames[frame_idx],
            x_coords=x_vals,
            y_coords=y_vals,
            visible_mask=vis_mask,
            color=color,
        )
    return frames


def _render_latent_canvas_video(
    canvas_heat: torch.Tensor,
    max_frames: int,
) -> np.ndarray:
    if canvas_heat.ndim != 3:
        raise ValueError(f"canvas_heat must be [F,H,W], got {tuple(canvas_heat.shape)}")
    heat_np = np.maximum(canvas_heat.detach().float().cpu().numpy(), 0.0)
    t_total = int(heat_np.shape[0])
    t_vis = t_total if int(max_frames) <= 0 else min(t_total, int(max_frames))
    if t_vis <= 0:
        h = int(max(heat_np.shape[1], 1))
        w = int(max(heat_np.shape[2], 1))
        return np.zeros((0, h, w, 3), dtype=np.uint8)

    clip_val = float(np.percentile(heat_np, 99.5))
    clip_val = max(clip_val, 1e-6)
    gamma = 0.6

    frames: List[np.ndarray] = []
    for frame_idx in range(t_vis):
        norm = np.clip(heat_np[frame_idx] / clip_val, 0.0, 1.0)
        norm = np.power(norm, gamma)
        red = (norm * 255.0).astype(np.uint8)
        green = (np.sqrt(norm) * 220.0).astype(np.uint8)
        blue = ((1.0 - norm) * 40.0).astype(np.uint8)
        rgb = np.stack([red, green, blue], axis=-1)
        frames.append(rgb)
    return np.stack(frames, axis=0)


def _expand_canvas_heat_to_decoded_t(
    canvas_heat: torch.Tensor,
    target_frames: int,
    temporal_ratio: int,
) -> torch.Tensor:
    if canvas_heat.ndim != 3:
        raise ValueError(f"canvas_heat must be [F,H,W], got {tuple(canvas_heat.shape)}")
    target_t = int(target_frames)
    if target_t <= 0:
        return canvas_heat[:0]
    t_ratio = max(int(temporal_ratio), 1)
    src_t = int(canvas_heat.shape[0])
    if src_t == 0:
        return canvas_heat[:0]

    # Wan temporal packing inverse approximation with model temporal ratio:
    # latent frame 0 -> decoded frame 0, latent frame 1.. -> `temporal_ratio` decoded frames each.
    first = canvas_heat[:1]
    rest = canvas_heat[1:]
    if rest.shape[0] > 0:
        expanded = torch.cat([first, rest.repeat_interleave(t_ratio, dim=0)], dim=0)
    else:
        expanded = first

    # No temporal interpolation/padding. Only truncate when caller requested fewer frames.
    if int(expanded.shape[0]) > target_t:
        return expanded[:target_t]
    return expanded


def _expand_canvas_rgb_to_pixel_grid_no_resample(
    canvas_rgb: np.ndarray,
    decoded_h: int,
    decoded_w: int,
    spatial_ratio: int,
) -> Optional[np.ndarray]:
    if canvas_rgb.ndim != 4 or canvas_rgb.shape[-1] != 3:
        raise ValueError(f"canvas_rgb must be [T,H,W,3], got {tuple(canvas_rgb.shape)}")
    h_lat = int(canvas_rgb.shape[1])
    w_lat = int(canvas_rgb.shape[2])
    if h_lat <= 0 or w_lat <= 0:
        return None

    ratio = max(int(spatial_ratio), 1)
    expected_h = h_lat * ratio
    expected_w = w_lat * ratio
    if expected_h != int(decoded_h) or expected_w != int(decoded_w):
        if (
            int(decoded_h) % h_lat == 0
            and int(decoded_w) % w_lat == 0
            and (int(decoded_h) // h_lat) == (int(decoded_w) // w_lat)
        ):
            ratio = int(decoded_h) // h_lat
            expected_h = h_lat * ratio
            expected_w = w_lat * ratio
        else:
            return None

    expanded = np.repeat(canvas_rgb, ratio, axis=1)
    expanded = np.repeat(expanded, ratio, axis=2)
    if int(expanded.shape[1]) != int(decoded_h) or int(expanded.shape[2]) != int(decoded_w):
        return None
    return expanded


def _overlay_canvas_on_decoded_video(
    decoded_clip: np.ndarray,
    canvas_clip: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    if decoded_clip.ndim != 4 or decoded_clip.shape[-1] != 3:
        raise ValueError(f"decoded_clip must be [T,H,W,3], got {tuple(decoded_clip.shape)}")
    if canvas_clip.ndim != 4 or canvas_clip.shape[-1] != 3:
        raise ValueError(f"canvas_clip must be [T,H,W,3], got {tuple(canvas_clip.shape)}")

    t = min(int(decoded_clip.shape[0]), int(canvas_clip.shape[0]))
    if t <= 0:
        return np.zeros_like(decoded_clip[:0])
    if (
        int(decoded_clip.shape[1]) != int(canvas_clip.shape[1])
        or int(decoded_clip.shape[2]) != int(canvas_clip.shape[2])
    ):
        raise ValueError(
            "decoded/canvas spatial shape mismatch for overlay: "
            f"decoded={tuple(decoded_clip.shape)} canvas={tuple(canvas_clip.shape)}"
        )

    base = decoded_clip[:t].astype(np.float32)
    heat = canvas_clip[:t].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    out = ((1.0 - a) * base + a * heat).clip(0.0, 255.0).astype(np.uint8)
    return out


def _track_point_hit_ratio_against_canvas(
    canvas_heat_aligned: torch.Tensor,
    tracks_px: torch.Tensor,
    visibility: torch.Tensor,
    decoded_h: int,
    decoded_w: int,
) -> Tuple[int, int, float, int]:
    """Measure how many visible track points land on active latent-canvas cells.

    This checks spatio-temporal alignment after temporal expansion by:
      1) mapping decoded-pixel track points back to latent grid cells
      2) testing whether those cells are active in expanded latent canvas heat

    Returns:
      - hit_count: visible points landing on non-zero canvas pixels
      - total_count: visible in-bound points evaluated
      - hit_ratio: hit_count / total_count
      - frames_eval: number of frames used for evaluation
    """
    if canvas_heat_aligned.ndim != 3:
        raise ValueError(
            f"canvas_heat_aligned must be [T,H_lat,W_lat], got {tuple(canvas_heat_aligned.shape)}"
        )
    if tracks_px.ndim != 3 or tracks_px.shape[-1] != 2:
        raise ValueError(f"tracks_px must be [T,P,2], got {tuple(tracks_px.shape)}")
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be [T,P], got {tuple(visibility.shape)}")

    t = min(
        int(canvas_heat_aligned.shape[0]),
        int(tracks_px.shape[0]),
        int(visibility.shape[0]),
    )
    if t <= 0:
        return 0, 0, 0.0, 0

    heat_np = canvas_heat_aligned[:t].detach().float().cpu().numpy()
    canvas_mask = heat_np > 0.0
    tracks_np = tracks_px[:t].detach().cpu().numpy()
    vis_np = visibility[:t].detach().cpu().numpy() > 0.5

    h_lat = int(canvas_mask.shape[1])
    w_lat = int(canvas_mask.shape[2])
    x = tracks_np[..., 0]
    y = tracks_np[..., 1]
    gx = np.floor(x / float(max(int(decoded_w), 1)) * float(max(w_lat - 1, 1))).astype(np.int64)
    gy = np.floor(y / float(max(int(decoded_h), 1)) * float(max(h_lat - 1, 1))).astype(np.int64)
    in_bounds = (gx >= 0) & (gx < w_lat) & (gy >= 0) & (gy < h_lat)
    valid = vis_np & in_bounds

    total_count = int(valid.sum())
    if total_count <= 0:
        return 0, 0, 0.0, t

    frame_idx = np.broadcast_to(np.arange(t, dtype=np.int64).reshape(t, 1), x.shape)
    hits = canvas_mask[frame_idx[valid], gy[valid], gx[valid]]
    hit_count = int(hits.sum())
    hit_ratio = float(hit_count / max(total_count, 1))
    return hit_count, total_count, hit_ratio, t


def _build_track_canvas_heat_for_vis(
    track_model: torch.nn.Module,
    track_condition: Dict[str, torch.Tensor],
    sample_idx: int,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    required_methods = (
        "_scatter_tracks_to_canvas",
        "_pack_track_to_wan_latent_time",
        "_align_track_latent_frames",
    )
    if not all(hasattr(track_model, name) for name in required_methods):
        return None

    tracks = track_condition["tracks"][sample_idx : sample_idx + 1].detach().to(
        device=device,
        dtype=torch.float32,
    )
    visibility = track_condition["visibility"][sample_idx : sample_idx + 1].detach().to(
        device=device,
        dtype=torch.float32,
    )
    point_mask = track_condition.get("point_mask", None)
    if point_mask is not None:
        point_mask_t = torch.as_tensor(point_mask).detach()
        if point_mask_t.ndim == 1:
            point_mask_t = point_mask_t.view(1, -1)
        elif point_mask_t.ndim == 2:
            point_mask_t = point_mask_t[
                min(sample_idx, point_mask_t.shape[0] - 1) : min(sample_idx, point_mask_t.shape[0] - 1) + 1
            ]
        else:
            raise ValueError(f"point_mask must be [P] or [B,P], got {tuple(point_mask_t.shape)}")
        point_mask = point_mask_t.to(device=device, dtype=torch.bool)
    is_normalized_hint = track_condition.get("is_normalized", None)
    if is_normalized_hint is not None:
        is_norm_t = torch.as_tensor(is_normalized_hint).detach()
        if is_norm_t.ndim == 0:
            is_norm_t = is_norm_t.view(1)
        else:
            is_norm_t = is_norm_t.view(-1)
            is_norm_t = is_norm_t[
                min(sample_idx, is_norm_t.shape[0] - 1) : min(sample_idx, is_norm_t.shape[0] - 1) + 1
            ]
        is_normalized_hint = is_norm_t.to(device=device)
    track_resolution = track_condition.get("track_resolution", None)
    if track_resolution is not None:
        tr = torch.as_tensor(track_resolution).detach().float()
        if tr.ndim == 1:
            if tr.numel() < 2:
                raise ValueError(f"track_resolution must contain [width,height], got {tuple(tr.shape)}")
            tr = tr[:2].view(1, 2)
        elif tr.ndim >= 2:
            tr = tr.view(-1, tr.shape[-1])
            if tr.shape[-1] < 2:
                raise ValueError(
                    f"track_resolution last dim must be >=2 for [width,height], got {tuple(tr.shape)}"
                )
            tr = tr[
                min(sample_idx, tr.shape[0] - 1) : min(sample_idx, tr.shape[0] - 1) + 1,
                :2,
            ]
        else:
            raise ValueError(f"Unexpected track_resolution shape: {tuple(tr.shape)}")
        track_resolution = tr.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        canvas = track_model._scatter_tracks_to_canvas(
            tracks=tracks,
            visibility=visibility,
            point_mask=point_mask,
            h_lat=int(latent_h),
            w_lat=int(latent_w),
            track_resolution=track_resolution,
            is_normalized_hint=is_normalized_hint,
        )
        packed = track_model._pack_track_to_wan_latent_time(canvas)
        packed = track_model._align_track_latent_frames(
            packed,
            target_frames=int(latent_frames),
        )
        canvas_heat = packed.abs().sum(dim=1)[0]  # [F,H,W]
    return canvas_heat.detach().cpu()


def _decode_latents_for_vis(
    latents_sample: torch.Tensor,
    vae: torch.nn.Module,
    decode_device: torch.device,
    decode_dtype: torch.dtype,
) -> np.ndarray:
    if latents_sample.ndim != 5 or latents_sample.shape[0] != 1:
        raise ValueError(
            f"latents_sample must be [1,C,F,H,W], got {tuple(latents_sample.shape)}"
        )

    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = decode_dtype if decode_device.type != "cpu" else torch.float32
    moved = (original_device != decode_device) or (original_dtype != target_dtype)
    if moved:
        vae.to(device=decode_device, dtype=target_dtype)

    try:
        with torch.no_grad():
            decoded = vae.decode(
                latents_sample.to(device=decode_device, dtype=target_dtype)
            ).sample
        decoded_uint8 = _video_tensor_to_uint8(decoded)
    finally:
        if moved:
            vae.to(device=original_device, dtype=original_dtype)
            if decode_device.type == "cuda" and original_device.type == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return decoded_uint8


def _export_track_debug_visualization(
    args: argparse.Namespace,
    accelerator: Accelerator,
    vae: torch.nn.Module,
    transformer3d_track: torch.nn.Module,
    latents: torch.Tensor,
    track_condition: Optional[Dict[str, torch.Tensor]],
    weight_dtype: torch.dtype,
    global_step: int,
) -> Optional[str]:
    if track_condition is None:
        return None
    if int(args.track_debug_vis_steps) <= 0:
        return None

    if latents.ndim != 5:
        raise ValueError(f"latents must be [B,C,F,H,W], got {tuple(latents.shape)}")

    bsz = int(latents.shape[0])
    sample_idx = min(max(int(args.track_debug_vis_sample_index), 0), max(bsz - 1, 0))
    latents_sample = latents[sample_idx : sample_idx + 1].detach()

    decode_device = accelerator.device if accelerator.device.type == "cuda" else latents_sample.device
    decoded_video = _decode_latents_for_vis(
        latents_sample=latents_sample,
        vae=vae,
        decode_device=decode_device,
        decode_dtype=weight_dtype,
    )

    decoded_h = int(decoded_video.shape[1])
    decoded_w = int(decoded_video.shape[2])
    tracks_px, visibility, (src_w, src_h), track_sample_idx = _extract_track_sample_for_vis(
        track_condition=track_condition,
        sample_index=sample_idx,
        max_points=int(args.track_debug_vis_max_points),
        default_w=decoded_w,
        default_h=decoded_h,
    )

    t_track = int(tracks_px.shape[0])
    t_decoded = int(decoded_video.shape[0])
    max_frames = int(args.track_debug_vis_max_frames)
    t_decoded_vis = t_decoded
    if max_frames > 0:
        t_decoded_vis = min(t_decoded_vis, max_frames)
    t_track_vis = min(t_track, t_decoded_vis)
    if t_decoded_vis <= 0 or t_track_vis <= 0:
        return None

    tracks_px = tracks_px[:t_track_vis]
    visibility = visibility[:t_track_vis]
    decoded_clip = decoded_video[:t_decoded_vis].copy()
    decoded_clip_for_track = decoded_clip[:t_track_vis].copy()
    pixel_track_clip = _render_track_points_video(
        tracks_px=tracks_px,
        visibility=visibility,
        canvas_h=max(src_h, 1),
        canvas_w=max(src_w, 1),
        max_frames=t_track_vis,
    )

    overlay_clip = decoded_clip_for_track.copy()
    overlay_tracks_px = tracks_px.clone()
    if src_w != decoded_w or src_h != decoded_h:
        overlay_tracks_px[..., 0] = overlay_tracks_px[..., 0] * (
            float(decoded_w) / float(max(src_w, 1))
        )
        overlay_tracks_px[..., 1] = overlay_tracks_px[..., 1] * (
            float(decoded_h) / float(max(src_h, 1))
        )
    tracks_np = overlay_tracks_px.detach().cpu().numpy()
    visibility_np = visibility.detach().cpu().numpy()
    for frame_idx in range(t_track_vis):
        overlay_clip[frame_idx] = _draw_visible_track_points(
            frame_rgb=overlay_clip[frame_idx],
            x_coords=tracks_np[frame_idx, :, 0],
            y_coords=tracks_np[frame_idx, :, 1],
            visible_mask=visibility_np[frame_idx] > 0.5,
        )

    track_model = accelerator.unwrap_model(transformer3d_track)
    canvas_heat = _build_track_canvas_heat_for_vis(
        track_model=track_model,
        track_condition=track_condition,
        sample_idx=track_sample_idx,
        latent_frames=int(latents_sample.shape[2]),
        latent_h=int(latents_sample.shape[3]),
        latent_w=int(latents_sample.shape[4]),
        device=latents_sample.device,
    )
    latent_canvas_clip = None
    canvas_heat_aligned = None
    decoded_grid_canvas_clip = None
    decoded_overlay_with_canvas_clip = None
    decoded_canvas_point_hits = 0
    decoded_canvas_point_total = 0
    decoded_canvas_point_hit_ratio = 0.0
    decoded_canvas_point_eval_frames = 0
    if canvas_heat is not None:
        latent_canvas_clip = _render_latent_canvas_video(
            canvas_heat=canvas_heat,
            max_frames=t_decoded_vis,
        )
        temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 4))
        spatial_ratio = int(getattr(vae, "spatial_compression_ratio", 8))
        canvas_heat_aligned = _expand_canvas_heat_to_decoded_t(
            canvas_heat=canvas_heat,
            target_frames=t_decoded_vis,
            temporal_ratio=temporal_ratio,
        )
        decoded_grid_canvas_lat_clip = _render_latent_canvas_video(
            canvas_heat=canvas_heat_aligned,
            max_frames=t_decoded_vis,
        )
        decoded_grid_canvas_clip = _expand_canvas_rgb_to_pixel_grid_no_resample(
            canvas_rgb=decoded_grid_canvas_lat_clip,
            decoded_h=decoded_h,
            decoded_w=decoded_w,
            spatial_ratio=spatial_ratio,
        )
        if decoded_grid_canvas_clip is not None:
            t_min = min(int(decoded_grid_canvas_clip.shape[0]), int(decoded_clip.shape[0]))
            decoded_for_canvas = decoded_clip[:t_min]
            decoded_grid_canvas_clip = decoded_grid_canvas_clip[:t_min]
            decoded_overlay_with_canvas_clip = _overlay_canvas_on_decoded_video(
                decoded_clip=decoded_for_canvas,
                canvas_clip=decoded_grid_canvas_clip,
                alpha=0.45,
            )
            (
                decoded_canvas_point_hits,
                decoded_canvas_point_total,
                decoded_canvas_point_hit_ratio,
                decoded_canvas_point_eval_frames,
            ) = _track_point_hit_ratio_against_canvas(
                canvas_heat_aligned=canvas_heat_aligned,
                tracks_px=overlay_tracks_px,
                visibility=visibility,
                decoded_h=decoded_h,
                decoded_w=decoded_w,
            )
        else:
            logger.warning(
                "Skipping decoded overlay with latent canvas at step=%d due non-integer spatial mapping "
                "(canvas_hw=%s decoded_hw=(%d,%d), spatial_ratio=%d).",
                int(global_step),
                tuple(int(v) for v in decoded_grid_canvas_lat_clip.shape[1:3]),
                int(decoded_h),
                int(decoded_w),
                int(spatial_ratio),
            )

    vis_root = os.path.join(
        args.output_dir_track,
        args.track_debug_vis_dir,
        f"step_{int(global_step):07d}",
    )
    os.makedirs(vis_root, exist_ok=True)
    _save_video_mp4_uint8(
        pixel_track_clip,
        os.path.join(vis_root, "pixel_track_hw.mp4"),
        fps=args.track_debug_vis_fps,
    )
    _save_video_mp4_uint8(
        decoded_clip,
        os.path.join(vis_root, "decoded_latent_video.mp4"),
        fps=args.track_debug_vis_fps,
    )
    _save_video_mp4_uint8(
        overlay_clip,
        os.path.join(vis_root, "decoded_overlay_with_pixel_track.mp4"),
        fps=args.track_debug_vis_fps,
    )
    if latent_canvas_clip is not None:
        _save_video_mp4_uint8(
            latent_canvas_clip,
            os.path.join(vis_root, "latent_hw_track_canvas.mp4"),
            fps=args.track_debug_vis_fps,
        )
    if decoded_grid_canvas_clip is not None:
        _save_video_mp4_uint8(
            decoded_grid_canvas_clip,
            os.path.join(vis_root, "decoded_hw_track_canvas_no_resample.mp4"),
            fps=args.track_debug_vis_fps,
        )
    if decoded_overlay_with_canvas_clip is not None:
        _save_video_mp4_uint8(
            decoded_overlay_with_canvas_clip,
            os.path.join(vis_root, "decoded_overlay_with_latent_canvas.mp4"),
            fps=args.track_debug_vis_fps,
        )

    summary = {
        "global_step": int(global_step),
        "sample_index_in_batch": int(track_sample_idx),
        "track_points_visualized": int(tracks_px.shape[1]),
        "track_frames_visualized": int(t_track_vis),
        "decoded_frames_visualized": int(t_decoded_vis),
        "source_track_resolution_wh": [int(src_w), int(src_h)],
        "latent_shape": [int(v) for v in latents_sample.shape[1:]],
        "decoded_shape": [int(v) for v in decoded_clip.shape],
    }
    if canvas_heat is not None:
        summary["latent_canvas_shape"] = [int(v) for v in canvas_heat.shape]
    if latent_canvas_clip is not None:
        summary["latent_canvas_video_shape"] = [int(v) for v in latent_canvas_clip.shape]
    if decoded_grid_canvas_clip is not None:
        summary["decoded_grid_canvas_video_shape"] = [int(v) for v in decoded_grid_canvas_clip.shape]
    temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 4))
    spatial_ratio = int(getattr(vae, "spatial_compression_ratio", 8))
    latent_frames = int(latents_sample.shape[2])
    latent_h = int(latents_sample.shape[3])
    latent_w = int(latents_sample.shape[4])
    expected_decoded_frames = 1 + max(latent_frames - 1, 0) * max(temporal_ratio, 1)
    summary["temporal_ratio"] = int(temporal_ratio)
    summary["spatial_ratio"] = int(spatial_ratio)
    summary["expected_decoded_frames_from_latent"] = int(expected_decoded_frames)
    summary["decoded_frames_match_expected"] = bool(int(t_decoded) == int(expected_decoded_frames))
    summary["decoded_h_over_latent_h"] = float(decoded_h / max(latent_h, 1))
    summary["decoded_w_over_latent_w"] = float(decoded_w / max(latent_w, 1))
    summary["decoded_spatial_matches_ratio"] = bool(
        int(decoded_h) == int(latent_h * max(spatial_ratio, 1))
        and int(decoded_w) == int(latent_w * max(spatial_ratio, 1))
    )
    summary["decoded_canvas_point_hits"] = int(decoded_canvas_point_hits)
    summary["decoded_canvas_point_total"] = int(decoded_canvas_point_total)
    summary["decoded_canvas_point_hit_ratio"] = float(decoded_canvas_point_hit_ratio)
    summary["decoded_canvas_point_eval_frames"] = int(decoded_canvas_point_eval_frames)
    with open(os.path.join(vis_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return vis_root


def _pad_precomputed_prompt(
    prompt_list: List[Optional[torch.Tensor]],
    mask_list: List[Optional[torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    valid_prompts = [p for p in prompt_list if p is not None]
    if len(valid_prompts) == 0:
        return None

    bsz = len(prompt_list)
    max_len = max(p.shape[0] for p in valid_prompts)
    d_model = valid_prompts[0].shape[1]
    embeds = torch.zeros((bsz, max_len, d_model), dtype=valid_prompts[0].dtype)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, prompt in enumerate(prompt_list):
        if prompt is None:
            continue
        cur_len = prompt.shape[0]
        embeds[i, :cur_len] = prompt
        cur_mask = mask_list[i]
        if cur_mask is not None:
            attention_mask[i, :cur_len] = cur_mask[:cur_len]
        else:
            attention_mask[i, :cur_len] = 1

    return {
        "prompt_embeds": embeds,
        "attention_mask": attention_mask,
    }


def _encode_uncond_prompt_embeds_track(
    tokenizer,
    text_encoder: torch.nn.Module,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same tokenization as training; empty string matches typical CFG unconditional."""
    with torch.no_grad():
        batch_enc = tokenizer(
            [""],
            padding="max_length",
            max_length=max_length,
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
        prompt_embeds = hidden[0, :valid_len].detach().float().cpu().clone()
        attention_mask = attn[0, :valid_len].detach().long().cpu().clone()
    return prompt_embeds, attention_mask


def _load_uncond_prompt_npz_track(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load null T5 features saved like dataset text_feature_wan_t5.npz."""
    data = np.load(path, allow_pickle=True)
    if "prompt_embeds" not in data:
        raise ValueError(f'{path} must contain key "prompt_embeds" (text_feature_wan_t5.npz layout).')
    pe = torch.as_tensor(np.array(data["prompt_embeds"]), dtype=torch.float32)
    if pe.ndim != 2:
        raise ValueError(f"prompt_embeds must be [L, D], got {tuple(pe.shape)}")
    if "attention_mask" in data:
        am = torch.as_tensor(np.array(data["attention_mask"]), dtype=torch.long).reshape(-1)
        n = min(int(pe.shape[0]), int(am.shape[0]))
        pe = pe[:n].contiguous()
        am = am[:n].contiguous()
    else:
        am = torch.ones((int(pe.shape[0]),), dtype=torch.long)
    return pe.cpu().clone(), am.cpu().clone()


def _apply_uncond_to_dropped_text_precomputed(
    embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    texts: List[str],
    uncond_embeds: torch.Tensor,
    uncond_mask: torch.Tensor,
) -> None:
    """In-place: rows with empty caption use unconditional T5 features, not file precomputes."""
    t_max = int(embeds.shape[1])
    le = int(uncond_embeds.shape[0])
    lt = min(le, t_max)
    for i, text in enumerate(texts):
        if text != "":
            continue
        embeds[i].zero_()
        embeds[i, :lt] = uncond_embeds[:lt].to(device=embeds.device, dtype=embeds.dtype)
        attention_mask[i].zero_()
        attention_mask[i, :lt] = uncond_mask[:lt].to(
            device=attention_mask.device, dtype=attention_mask.dtype
        )


def _pad_precomputed_clip_feature(
    clip_list: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    valid_clip = [c for c in clip_list if c is not None]
    if len(valid_clip) == 0:
        return None

    bsz = len(clip_list)
    max_tokens = max(c.shape[0] for c in valid_clip)
    d_model = valid_clip[0].shape[1]
    clip_feature = torch.zeros((bsz, max_tokens, d_model), dtype=valid_clip[0].dtype)
    for i, c in enumerate(clip_list):
        if c is None:
            continue
        clip_feature[i, : c.shape[0]] = c
    return clip_feature


def _resize_mask_like_pipeline(
    mask: torch.Tensor,
    latent: torch.Tensor,
    process_first_frame_only: bool = True,
) -> torch.Tensor:
    latent_size = latent.size()
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
    return resized_mask


def _encode_masked_video_like_pipeline(
    masked_video: torch.Tensor,
    vae,
    device: torch.device,
) -> torch.Tensor:
    # Keep the same micro-batch encode style used in pipeline.prepare_mask_latents.
    masked_video = masked_video.to(device=device, dtype=vae.dtype)
    bs = 1
    chunks = []
    for i in range(0, masked_video.shape[0], bs):
        mb = masked_video[i : i + bs]
        mb_latents = vae.encode(mb)[0].mode()
        chunks.append(mb_latents)
    return torch.cat(chunks, dim=0)


def _checkpoint_scan_root_track(output_dir: str, checkpoint_dir: Optional[str]) -> str:
    return checkpoint_dir if checkpoint_dir else output_dir


def _resolve_resume_checkpoint_track(
    resume: str,
    output_dir: str,
    checkpoint_dir: Optional[str],
) -> str:
    scan_root = _checkpoint_scan_root_track(output_dir, checkpoint_dir)
    if resume.strip().lower() == "latest":
        best_n = -1
        best_path: Optional[str] = None
        if os.path.isdir(scan_root):
            for name in os.listdir(scan_root):
                m = re.fullmatch(r"checkpoint-(\d+)", name)
                if not m:
                    continue
                n = int(m.group(1))
                if n > best_n:
                    best_n = n
                    best_path = os.path.join(scan_root, name)
        if best_path is None:
            raise ValueError(
                f'resume_from_checkpoint_track="latest" but no checkpoint-* directory found under "{scan_root}"'
            )
        return best_path

    resume_expanded = os.path.expanduser(resume)
    if os.path.isdir(resume_expanded):
        return os.path.abspath(resume_expanded)
    candidate = os.path.join(scan_root, resume_expanded)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    raise ValueError(
        f'--resume_from_checkpoint_track path not found: "{resume}" '
        f'(also tried "{candidate}")'
    )


def _read_global_step_from_checkpoint_track(
    checkpoint_path: str,
    gradient_accumulation_steps: int,
    accelerator_step: int,
) -> int:
    sidecar = os.path.join(checkpoint_path, "trainer_state_track.json")
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data["global_step"])
    m = re.search(r"checkpoint-(\d+)$", checkpoint_path.rstrip(os.sep))
    if m:
        return int(m.group(1))
    return max(0, accelerator_step // max(1, gradient_accumulation_steps))


def _write_trainer_state_track(checkpoint_path: str, global_step: int) -> None:
    sidecar = os.path.join(checkpoint_path, "trainer_state_track.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step}, f)


def _load_model_only_checkpoint_track(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, List[str]]:
    model_safetensors = os.path.join(checkpoint_path, "model.safetensors")
    model_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
    legacy_model_bin = os.path.join(checkpoint_path, "model.bin")

    if os.path.isfile(model_safetensors):
        from safetensors.torch import load_file

        state_dict = load_file(model_safetensors, device="cpu")
    elif os.path.isfile(model_bin):
        state_dict = torch.load(model_bin, map_location="cpu")
    elif os.path.isfile(legacy_model_bin):
        state_dict = torch.load(legacy_model_bin, map_location="cpu")
    else:
        raise ValueError(
            f'No model weights found in checkpoint "{checkpoint_path}". '
            "Expected one of: model.safetensors, pytorch_model.bin, model.bin"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 <= float(args.text_drop_ratio_track) <= 1.0):
        raise ValueError("--text_drop_ratio_track must be in [0, 1].")
    if not (0.0 <= float(args.first_frame_condition_drop_prob_track) <= 1.0):
        raise ValueError("--first_frame_condition_drop_prob_track must be in [0, 1].")
    if int(args.validation_subset_size_track) < 0:
        raise ValueError("--validation_subset_size_track must be >= 0.")
    if int(args.validation_max_batches_track) < 0:
        raise ValueError("--validation_max_batches_track must be >= 0.")
    if int(args.validation_track_max_points_track) < -1:
        raise ValueError("--validation_track_max_points_track must be >= -1.")
    if int(args.track_debug_vis_steps) < 0:
        raise ValueError("--track_debug_vis_steps must be >= 0.")
    if int(args.track_debug_vis_fps) <= 0:
        raise ValueError("--track_debug_vis_fps must be > 0.")
    if args.track_patch_init_gain is None:
        args.track_patch_init_gain = (
            float(args.track_patch_init_alpha) if args.track_patch_init_alpha is not None else 1.0
        )
    if args.track_patch_init_alpha is not None:
        if not math.isfinite(float(args.track_patch_init_alpha)):
            raise ValueError("--track_patch_init_alpha must be finite.")
        if float(args.track_patch_init_alpha) < 0.0:
            raise ValueError("--track_patch_init_alpha must be >= 0.")
    if not math.isfinite(float(args.track_patch_init_gain)):
        raise ValueError("--track_patch_init_gain must be finite.")
    if float(args.track_patch_init_gain) < 0.0:
        raise ValueError("--track_patch_init_gain must be >= 0.")
    if not math.isfinite(float(args.track_latent_scale)):
        raise ValueError("--track_latent_scale must be finite.")
    if float(args.track_latent_scale) < 0.0:
        raise ValueError("--track_latent_scale must be >= 0.")
    if not math.isfinite(float(args.track_init_noise_scale)):
        raise ValueError("--track_init_noise_scale must be finite.")
    if float(args.track_init_noise_scale) < 0.0:
        raise ValueError("--track_init_noise_scale must be >= 0.")
    if args.resume_from_checkpoint_track and args.init_model_from_checkpoint_track:
        raise ValueError(
            "--resume_from_checkpoint_track and --init_model_from_checkpoint_track are mutually exclusive."
        )
    root_map = _parse_root_map(args)
    os.makedirs(args.output_dir_track, exist_ok=True)
    if args.checkpoint_dir_track:
        os.makedirs(args.checkpoint_dir_track, exist_ok=True)
    if (not args.dummy_data_track) and (args.train_data_meta_track is None):
        raise ValueError("--train_data_meta_track is required unless --dummy_data_track is set.")
    if (not args.dummy_data_track) and args.train_data_meta_track is not None:
        detected_source_media = _sniff_metadata_source_media(args.train_data_meta_track)
        if detected_source_media == "latent" and args.input_mode_track != "latent":
            logger.warning(
                "Detected latent metadata from %s; overriding --input_mode_track=%s -> latent",
                args.train_data_meta_track,
                args.input_mode_track,
            )
            args.input_mode_track = "latent"
    if args.dummy_data_track and args.input_mode_track != "latent":
        logger.warning("dummy_data_track only supports latent tensors. Overriding input_mode_track=latent.")
        args.input_mode_track = "latent"
    if int(args.track_debug_vis_steps) > 0 and (not args.use_track_condition):
        logger.warning(
            "track_debug_vis_steps is enabled but --use_track_condition is off; debug visualization will be skipped."
        )
    if int(args.track_debug_vis_steps) > 0 and args.track_condition_mode != "track_head":
        logger.warning(
            "track_debug_vis_steps currently visualizes the learned track_head canvas only; "
            "it will be skipped for track_condition_mode=%s.",
            args.track_condition_mode,
        )
    is_latent_mode = args.input_mode_track == "latent"
    if args.track_condition_mode == "wan_move":
        if not args.use_track_condition:
            raise ValueError("--track_condition_mode=wan_move requires --use_track_condition.")
        if not is_latent_mode:
            raise ValueError("--track_condition_mode=wan_move currently requires --input_mode_track=latent.")
        if args.train_mode == "normal" or (not args.use_first_frame_condition_track):
            raise ValueError(
                "--track_condition_mode=wan_move requires inpaint/i2v first-frame conditioning "
                "(--train_mode != normal and --use_first_frame_condition_track)."
            )

    project_config = ProjectConfiguration(
        project_dir=args.output_dir_track,
        logging_dir=os.path.join(args.output_dir_track, args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )
    if args.seed is not None:
        set_seed(args.seed)

    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).eval()

    clip_image_encoder = None
    if (not is_latent_mode) and args.train_mode != "normal":
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(
                args.pretrained_model_name_or_path,
                config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
            )
        ).eval()

    transformer_additional_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    if args.track_condition_mode == "track_head" and args.track_head_hidden_dim is not None:
        if int(args.track_head_hidden_dim) <= 0:
            raise ValueError("--track_head_hidden_dim must be > 0 when provided.")
        transformer_additional_kwargs["track_head_hidden_dim"] = int(args.track_head_hidden_dim)
    if args.track_condition_mode == "track_head":
        transformer_additional_kwargs["track_latent_scale"] = float(args.track_latent_scale)

    transformer_cls = WanTransformer3DModelTrack if args.track_condition_mode == "track_head" else WanTransformer3DModel
    transformer3d_track = transformer_cls.from_pretrained(
        os.path.join(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=transformer_additional_kwargs,
    ).to(weight_dtype)
    if args.init_model_from_checkpoint_track:
        init_path = _resolve_resume_checkpoint_track(
            args.init_model_from_checkpoint_track,
            args.output_dir_track,
            args.checkpoint_dir_track,
        )
        load_info = _load_model_only_checkpoint_track(transformer3d_track, init_path)
        if accelerator.is_main_process:
            miss_cnt = len(load_info["missing"])
            unexp_cnt = len(load_info["unexpected"])
            msg = (
                f"Initialized model-only from {init_path} "
                f"(missing={miss_cnt}, unexpected={unexp_cnt}). "
                "Optimizer/lr scheduler/global_step will start from scratch."
            )
            logger.info(msg)
            accelerator.print(msg)
            if miss_cnt > 0:
                miss_preview = ", ".join(load_info["missing"][:10])
                logger.warning(
                    "Model init missing keys (first 10/%d): %s",
                    miss_cnt,
                    miss_preview,
                )
            if unexp_cnt > 0:
                unexp_preview = ", ".join(load_info["unexpected"][:10])
                logger.warning(
                    "Model init unexpected keys (first 10/%d): %s",
                    unexp_cnt,
                    unexp_preview,
                )
    if (
        args.use_track_condition
        and args.track_condition_mode == "track_head"
        and args.apply_track_patch_embed_init_track
    ):
        if not hasattr(transformer3d_track, "apply_post_load_i2v_track_patch_embed_init"):
            raise ValueError(
                "transformer3d_track does not support post-load patch_embed track init: "
                "missing apply_post_load_i2v_track_patch_embed_init()."
            )
        patch_init_info = transformer3d_track.apply_post_load_i2v_track_patch_embed_init(
            latent_channels=int(getattr(transformer3d_track, "track_concat_channels", 0)),
            mode=str(args.track_patch_init_mode),
            gain=float(args.track_patch_init_gain),
            add_noise=bool(args.add_track_init_noise),
            noise_scale=float(args.track_init_noise_scale),
            base_input_order="denoise_mask_first_frame",
        )
        if accelerator.is_main_process:
            patch_init_msg = (
                "Track patch_embedding scaled-duplicate init applied: "
                f"old_shape={patch_init_info['old_weight_shape']} -> "
                f"new_shape={patch_init_info['new_weight_shape']}, "
                f"C={patch_init_info['latent_channels']}, "
                f"mode={patch_init_info['mode']}, "
                f"gain={patch_init_info['gain']:.6f}, "
                f"add_noise={patch_init_info['add_noise']}, "
                f"noise_scale={patch_init_info['noise_scale']:.6f}, "
                f"source={patch_init_info['source_block']}, "
                f"base_input_order={patch_init_info['base_input_order']} "
                f"({patch_init_info['base_layout']})"
            )
            logger.info(patch_init_msg)
            accelerator.print(patch_init_msg)
    elif accelerator.is_main_process:
        skip_patch_init_msg = (
            "Track patch_embedding scaled-duplicate init skipped: "
            f"use_track_condition={bool(args.use_track_condition)}, "
            f"track_condition_mode={args.track_condition_mode}, "
            f"apply_track_patch_embed_init_track={bool(args.apply_track_patch_embed_init_track)}"
        )
        logger.info(skip_patch_init_msg)
        accelerator.print(skip_patch_init_msg)

    if (
        accelerator.is_main_process
        and args.track_condition_mode == "track_head"
        and hasattr(transformer3d_track, "get_patch_embedding_track_init_stats")
    ):
        init_stats = transformer3d_track.get_patch_embedding_track_init_stats()
        init_msg = (
            "Track patch_embedding init check: "
            f"old_in={int(init_stats['old_in_channels'])}, "
            f"new_in={int(init_stats['new_in_channels'])}, "
            f"added_in={int(init_stats['added_in_channels'])}, "
            f"mean={init_stats['added_weight_mean']:.6e}, "
            f"std={init_stats['added_weight_std']:.6e}, "
            f"abs_mean={init_stats['added_weight_abs_mean']:.6e}, "
            f"min={init_stats['added_weight_min']:.6e}, "
            f"max={init_stats['added_weight_max']:.6e}"
        )
        logger.info(init_msg)
        accelerator.print(init_msg)

    # Keep checkpointing controllable from CLI while preserving default behavior.
    # Prefer HF-style API when available for compatibility with DDP/Accelerate wrapping.
    if args.gradient_checkpointing:
        if hasattr(transformer3d_track, "gradient_checkpointing_enable"):
            transformer3d_track.gradient_checkpointing_enable()
        elif hasattr(transformer3d_track, "_set_gradient_checkpointing"):
            transformer3d_track._set_gradient_checkpointing(enable=True)
    else:
        if hasattr(transformer3d_track, "gradient_checkpointing_disable"):
            transformer3d_track.gradient_checkpointing_disable()
        elif hasattr(transformer3d_track, "_set_gradient_checkpointing"):
            transformer3d_track._set_gradient_checkpointing(enable=False)

    # Some transformer implementations use `use_cache` and it can conflict with checkpointing.
    # Turn it off defensively only when present and checkpointing is enabled.
    if args.gradient_checkpointing and hasattr(transformer3d_track, "config"):
        try:
            use_cache = getattr(transformer3d_track.config, "use_cache", None)
            if use_cache:
                logger.warning("Disabling use_cache because gradient checkpointing is enabled.")
                transformer3d_track.config.use_cache = False
        except Exception:
            # Ignore config mutation failures for immutable config objects.
            pass

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)
    transformer3d_track.requires_grad_(False)
    transformer3d_track.train()

    trainable_params = []
    trainable_named_params = []
    for name, param in transformer3d_track.named_parameters():
        if any(module_name in name for module_name in args.trainable_modules_track):
            trainable_params.append(param)
            trainable_named_params.append((name, param))
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters matched --trainable_modules_track")
    selected_trainable_named_params = []
    early_block_limit = int(args.train_early_blocks_track)
    for name, param in trainable_named_params:
        if early_block_limit < 0:
            selected_trainable_named_params.append((name, param))
            continue
        if _is_new_track_layer_param(name, args.track_condition_mode):
            selected_trainable_named_params.append((name, param))
            continue
        block_idx = _extract_block_index(name)
        if block_idx is not None and block_idx < early_block_limit:
            selected_trainable_named_params.append((name, param))
    if len(selected_trainable_named_params) == 0:
        raise ValueError(
            "No trainable parameters selected after applying --train_early_blocks_track and "
            "--trainable_modules_track."
        )
    trainable_named_params = selected_trainable_named_params
    trainable_params = [p for _, p in trainable_named_params]
    trainable_name_set = {name for name, _ in trainable_named_params}
    new_params_only_steps_track = max(0, int(args.new_params_only_steps_track))
    new_only_trainable_name_set = set()
    for _, param in trainable_named_params:
        param.requires_grad = False
    if new_params_only_steps_track > 0:
        for name, param in trainable_named_params:
            if _is_new_track_layer_param(name, args.track_condition_mode):
                param.requires_grad = True
                new_only_trainable_name_set.add(name)
        if len(new_only_trainable_name_set) == 0:
            for _, param in trainable_named_params:
                param.requires_grad = True
            new_params_only_steps_track = 0
            logger.warning(
                "new_params_only_steps_track requested, but no newly-initialized params are included by "
                "--trainable_modules_track. Falling back to full training from step 0."
            )
    else:
        for _, param in trainable_named_params:
            param.requires_grad = True
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in transformer3d_track.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        active_trainable_param_count = sum(
            p.numel() for _, p in trainable_named_params if p.requires_grad
        )
        param_msg = (
            "Transformer params total="
            f"{total_params} ({total_params / 1e6:.2f}M), "
            f"trainable_candidates={trainable_param_count} ({trainable_param_count / 1e6:.2f}M), "
            f"active_now={active_trainable_param_count} ({active_trainable_param_count / 1e6:.2f}M), "
            f"ratio={float(trainable_param_count) / float(max(total_params, 1)):.4f}"
        )
        logger.info(param_msg)
        accelerator.print(param_msg)
        selected_early_block_indices = sorted(
            {
                idx
                for name, _ in trainable_named_params
                for idx in [_extract_block_index(name)]
                if idx is not None
            }
        )
        selection_msg = (
            f"train_early_blocks_track={early_block_limit}, "
            f"selected_early_blocks={selected_early_block_indices}, "
            f"new_layer_param_tensors={sum(1 for name, _ in trainable_named_params if _is_new_track_layer_param(name, args.track_condition_mode))}, "
            f"base_param_tensors={sum(1 for name, _ in trainable_named_params if not _is_new_track_layer_param(name, args.track_condition_mode))}"
        )
        logger.info(selection_msg)
        accelerator.print(selection_msg)
        if new_params_only_steps_track > 0:
            phase_msg = (
                f"new_params_only phase enabled for first {new_params_only_steps_track} steps; "
                f"new_param_tensors={len(new_only_trainable_name_set)}"
            )
            logger.info(phase_msg)
            accelerator.print(phase_msg)

    if args.dummy_data_track:
        train_dataset = DummyTrackLatentDataset(
            length=args.dummy_length_track,
            latent_shape=(
                args.dummy_latent_channels_track,
                args.dummy_latent_frames_track,
                args.dummy_latent_h_track,
                args.dummy_latent_w_track,
            ),
            n_frames=args.dummy_n_frames_track,
            n_points=args.dummy_n_points_track,
            text="dummy track prompt",
        )
    elif args.input_mode_track == "latent":
        train_dataset = ImageVideoLatentTrackDataset(
            args.train_data_meta_track,
            args.train_data_dir,
            text_drop_ratio=float(args.text_drop_ratio_track),
            track_condition_key=args.track_condition_key,
            latent_file_key=args.latent_file_key_track,
            first_frame_vae_latent_file_key=args.first_frame_vae_latent_file_key_track,
            root_map=root_map,
            root_id_key=args.train_data_root_id_key_track,
        )
    else:
        train_dataset = ImageVideoDatasetTrack(
            args.train_data_meta_track,
            args.train_data_dir,
            video_sample_size=args.video_sample_size,
            video_sample_stride=args.video_sample_stride,
            video_sample_n_frames=args.video_sample_n_frames,
            image_sample_size=args.image_sample_size,
            video_repeat=0,
            text_drop_ratio=float(args.text_drop_ratio_track),
            enable_bucket=False,
            enable_inpaint=args.train_mode != "normal",
            track_condition_key=args.track_condition_key,
            root_map=root_map,
            root_id_key=args.train_data_root_id_key_track,
        )

    if accelerator.is_main_process and (not args.dummy_data_track):
        tdr_msg = (
            f"text_drop_ratio_track={float(args.text_drop_ratio_track)} "
            "(validation text_drop_ratio=0)"
        )
        logger.info(tdr_msg)
        accelerator.print(tdr_msg)
        ffdr_msg = (
            f"first_frame_condition_drop_prob_track={float(args.first_frame_condition_drop_prob_track)} "
            "(train only)"
        )
        logger.info(ffdr_msg)
        accelerator.print(ffdr_msg)

    uncond_pe_cpu: Optional[torch.Tensor] = None
    uncond_am_cpu: Optional[torch.Tensor] = None
    apply_text_drop_precomputed = (
        args.input_mode_track == "latent"
        and (not args.no_apply_text_dropout_to_precomputed_track)
    )
    if apply_text_drop_precomputed:
        if args.precomputed_uncond_text_npz_track:
            npz_path = os.path.expanduser(args.precomputed_uncond_text_npz_track)
            if not os.path.isfile(npz_path):
                raise FileNotFoundError(
                    f"--precomputed_uncond_text_npz_track not found: {npz_path}"
                )
            uncond_pe_cpu, uncond_am_cpu = _load_uncond_prompt_npz_track(npz_path)
            if accelerator.is_main_process:
                logger.info(
                    "Text-drop + precomputed: loaded null T5 embeds from %s shape=%s",
                    npz_path,
                    tuple(uncond_pe_cpu.shape),
                )
        else:
            enc_dev = accelerator.device
            text_encoder.to(enc_dev, dtype=weight_dtype)
            uncond_pe_cpu, uncond_am_cpu = _encode_uncond_prompt_embeds_track(
                tokenizer,
                text_encoder,
                args.tokenizer_max_length,
                enc_dev,
            )
            text_encoder.to("cpu")
            if accelerator.is_main_process:
                logger.info(
                    "Text-drop + precomputed: encoded null T5 embeds at startup shape=%s",
                    tuple(uncond_pe_cpu.shape),
                )

    def _make_collate_fn(is_validation: bool = False):
        def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
            texts = [example["text"] for example in examples]
            batch = {"text": texts}

            if args.input_mode_track == "latent":
                latents = torch.stack([example["latents"] for example in examples], dim=0)
                batch["latents"] = latents
                norm_h = args.track_normalize_height
                norm_w = args.track_normalize_width
                first_frames = [example.get("first_frame_pixel_values", None) for example in examples]
                if all(frame is not None for frame in first_frames):
                    batch["first_frame_pixel_values"] = torch.stack(first_frames, dim=0)
                first_frame_vae_latents = [example.get("first_frame_vae_latent", None) for example in examples]
                if all(latent is not None for latent in first_frame_vae_latents):
                    batch["first_frame_vae_latent"] = torch.stack(first_frame_vae_latents, dim=0)

                precomputed_prompt = _pad_precomputed_prompt(
                    [example.get("precomputed_prompt_embeds", None) for example in examples],
                    [example.get("precomputed_attention_mask", None) for example in examples],
                )
                if precomputed_prompt is not None:
                    batch["precomputed_prompt_embeds"] = precomputed_prompt["prompt_embeds"]
                    batch["precomputed_attention_mask"] = precomputed_prompt["attention_mask"]
                    if uncond_pe_cpu is not None:
                        _apply_uncond_to_dropped_text_precomputed(
                            batch["precomputed_prompt_embeds"],
                            batch["precomputed_attention_mask"],
                            texts,
                            uncond_pe_cpu,
                            uncond_am_cpu,
                        )

                precomputed_clip_feature = _pad_precomputed_clip_feature(
                    [example.get("precomputed_clip_feature", None) for example in examples]
                )
                if precomputed_clip_feature is not None:
                    batch["precomputed_clip_feature"] = precomputed_clip_feature
            else:
                pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
                batch["pixel_values"] = pixel_values
                norm_h = pixel_values.shape[-2]
                norm_w = pixel_values.shape[-1]

                if args.train_mode != "normal":
                    batch["mask_pixel_values"] = torch.stack([example["mask_pixel_values"] for example in examples], dim=0)
                    batch["mask"] = torch.stack([example["mask"] for example in examples], dim=0)
                    batch["clip_pixel_values"] = torch.stack([example["clip_pixel_values"] for example in examples], dim=0)

            if is_validation:
                val_track_max_points = int(args.validation_track_max_points_track)
                track_max_points = val_track_max_points if val_track_max_points > 0 else int(args.track_max_points)
                random_points_min = 0
                random_points_max = 0
                random_points = False
                track_drop_prob = 0.0
            else:
                track_max_points = int(args.track_max_points)
                random_points_min = int(args.track_random_points_min)
                random_points_max = int(args.track_random_points_max)
                random_points = True
                track_drop_prob = float(args.track_condition_drop_prob)

            track_batch = [example.get(args.track_condition_key, None) for example in examples]
            track_condition = _pad_track_condition(
                track_batch,
                max_points=track_max_points,
                random_points_min=random_points_min,
                random_points_max=random_points_max,
                sort_selected_indices=args.track_sort_selected_indices,
                point_id_mode=args.track_point_id_mode,
                normalize=args.track_normalize,
                h=norm_h,
                w=norm_w,
                random_points=random_points,
            )
            track_condition = _apply_track_condition_dropout(
                track_condition,
                drop_prob=track_drop_prob,
            )
            batch["track_condition"] = track_condition
            return batch

        return collate_fn

    train_collate_fn = _make_collate_fn(is_validation=False)
    val_collate_fn = _make_collate_fn(is_validation=True)

    if args.input_mode_track == "latent":
        batch_sampler = torch.utils.data.BatchSampler(
            RandomSampler(train_dataset),
            batch_size=args.train_batch_size,
            drop_last=True,
        )
    else:
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset), train_dataset, args.train_batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=train_collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    val_dataloader = None
    if args.validation_steps_track > 0 and args.val_data_meta_track:
        val_data_dir = args.val_data_dir_track if args.val_data_dir_track else args.train_data_dir
        if args.input_mode_track == "latent":
            val_dataset = ImageVideoLatentTrackDataset(
                args.val_data_meta_track,
                val_data_dir,
                text_drop_ratio=0.0,
                track_condition_key=args.track_condition_key,
                latent_file_key=args.latent_file_key_track,
                first_frame_vae_latent_file_key=args.first_frame_vae_latent_file_key_track,
                root_map=root_map,
                root_id_key=args.train_data_root_id_key_track,
            )
            val_full_len = len(val_dataset)
            if int(args.validation_subset_size_track) > 0:
                val_subset_indices = _make_fixed_subset_indices(
                    val_full_len,
                    int(args.validation_subset_size_track),
                    int(args.validation_subset_seed_track),
                )
                val_dataset = Subset(val_dataset, val_subset_indices)
                if accelerator.is_main_process:
                    subset_msg = (
                        f"validation fixed subset: size={len(val_dataset)}/{val_full_len}, "
                        f"seed={int(args.validation_subset_seed_track)}, "
                        f"first_indices={val_subset_indices[:8]}"
                    )
                    logger.info(subset_msg)
                    accelerator.print(subset_msg)
            val_batch_sampler = torch.utils.data.BatchSampler(
                SequentialSampler(val_dataset),
                batch_size=args.train_batch_size,
                drop_last=False,
            )
        else:
            val_dataset = ImageVideoDatasetTrack(
                args.val_data_meta_track,
                val_data_dir,
                video_sample_size=args.video_sample_size,
                video_sample_stride=args.video_sample_stride,
                video_sample_n_frames=args.video_sample_n_frames,
                image_sample_size=args.image_sample_size,
                enable_bucket=False,
                enable_inpaint=args.train_mode != "normal",
                track_condition_key=args.track_condition_key,
                root_map=root_map,
                root_id_key=args.train_data_root_id_key_track,
            )
            if int(args.validation_subset_size_track) > 0 and accelerator.is_main_process:
                logger.warning(
                    "validation_subset_size_track is currently applied only in latent mode; "
                    "video-mode validation will use the metadata prefix limited by validation_max_batches_track."
                )
            val_batch_sampler = ImageVideoSampler(
                SequentialSampler(val_dataset),
                val_dataset,
                args.train_batch_size,
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=val_collate_fn,
            num_workers=args.dataloader_num_workers,
            persistent_workers=args.dataloader_num_workers > 0,
        )
    elif args.validation_steps_track > 0 and accelerator.is_main_process:
        logger.warning(
            "validation_steps_track is set but val_data_meta_track is missing. Validation logging is disabled."
        )

    new_track_layers_lr = float(
        args.new_track_layers_lr if args.new_track_layers_lr is not None else args.learning_rate
    )
    early_blocks_lr = float(
        args.early_blocks_lr if args.early_blocks_lr is not None else args.learning_rate
    )
    new_layer_trainable_params = [
        p for name, p in trainable_named_params if _is_new_track_layer_param(name, args.track_condition_mode)
    ]
    base_trainable_params = [
        p for name, p in trainable_named_params if not _is_new_track_layer_param(name, args.track_condition_mode)
    ]
    optimizer_param_groups = []
    if len(new_layer_trainable_params) > 0:
        optimizer_param_groups.append(
            {"params": new_layer_trainable_params, "lr": new_track_layers_lr}
        )
    if len(base_trainable_params) > 0:
        optimizer_param_groups.append(
            {"params": base_trainable_params, "lr": early_blocks_lr}
        )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    if accelerator.is_main_process:
        optimizer_msg = (
            f"optimizer groups: new_layers={len(new_layer_trainable_params)} tensors @ lr={new_track_layers_lr:.2e}, "
            f"base_blocks={len(base_trainable_params)} tensors @ lr={early_blocks_lr:.2e}, "
            f"weight_decay={args.adam_weight_decay:.2e}"
        )
        logger.info(optimizer_msg)
        accelerator.print(optimizer_msg)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * max(1, len(train_dataloader))
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    (
        transformer3d_track,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(transformer3d_track, optimizer, train_dataloader, lr_scheduler)
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
    text_encoder_on_gpu = False
    if is_latent_mode:
        # Latent mode usually relies on precomputed latents/prompt/clip features.
        # Keep auxiliary encoders off GPU by default to preserve VRAM.
        if args.use_first_frame_condition_track and args.train_mode != "normal":
            vae.to(accelerator.device, dtype=weight_dtype)
        else:
            vae.to("cpu")
        text_encoder.to("cpu")
        if clip_image_encoder is not None:
            clip_image_encoder.to("cpu")
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_on_gpu = True
        if clip_image_encoder is not None:
            clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    image_processor = None
    mask_processor = None
    if args.use_first_frame_condition_track and args.train_mode != "normal":
        image_processor = VaeImageProcessor(vae_scale_factor=vae.config.spatial_compression_ratio)
        mask_processor = VaeImageProcessor(
            vae_scale_factor=vae.config.spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
    wan_move_temporal_stride = int(getattr(vae.config, "temporal_compression_ratio", 4))
    if accelerator.is_main_process:
        mode_msg = (
            f"track_condition_mode={args.track_condition_mode}, "
            f"vae_temporal_compression_ratio={wan_move_temporal_stride}"
        )
        logger.info(mode_msg)
        accelerator.print(mode_msg)

    ckpt_root = _checkpoint_scan_root_track(args.output_dir_track, args.checkpoint_dir_track)
    global_step = 0
    if args.resume_from_checkpoint_track:
        resume_path = _resolve_resume_checkpoint_track(
            args.resume_from_checkpoint_track,
            args.output_dir_track,
            args.checkpoint_dir_track,
        )
        accelerator.load_state(resume_path)
        global_step = _read_global_step_from_checkpoint_track(
            resume_path,
            args.gradient_accumulation_steps,
            accelerator.step,
        )
        if accelerator.is_main_process:
            msg = f"Resumed from {resume_path} (global_step={global_step})"
            logger.info(msg)
            accelerator.print(msg)

    if accelerator.is_main_process:
        gc_enabled = bool(getattr(transformer3d_track, "gradient_checkpointing", False))
        gc_msg = (
            f"Gradient checkpointing requested={args.gradient_checkpointing}, "
            f"effective={gc_enabled}"
        )
        logger.info(gc_msg)
        accelerator.print(gc_msg)
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
        init_kwargs = {}
        if args.wandb_run_name_track and args.report_to in ("wandb", "all"):
            init_kwargs["wandb"] = {"name": args.wandb_run_name_track}
        if init_kwargs:
            accelerator.init_trackers(
                args.tracker_project_name_track,
                config=tracker_config,
                init_kwargs=init_kwargs,
            )
        else:
            accelerator.init_trackers(args.tracker_project_name_track, config=tracker_config)

    new_only_phase_active = None
    patch_new_grad_mask_handle = None
    def _set_trainability_phase(new_only: bool) -> None:
        nonlocal new_only_phase_active
        nonlocal patch_new_grad_mask_handle
        if (new_only_phase_active is not None) and (new_only == new_only_phase_active):
            return
        unwrapped_model = accelerator.unwrap_model(transformer3d_track)
        for name, param in unwrapped_model.named_parameters():
            if name not in trainable_name_set:
                continue
            param.requires_grad = (name in new_only_trainable_name_set) if new_only else True

        if patch_new_grad_mask_handle is not None:
            patch_new_grad_mask_handle.remove()
            patch_new_grad_mask_handle = None

        if new_only and ("patch_embedding.weight" in new_only_trainable_name_set):
            patch_w = getattr(getattr(unwrapped_model, "patch_embedding", None), "weight", None)
            track_concat_channels_local = int(getattr(unwrapped_model, "track_concat_channels", 0))
            if (
                patch_w is not None
                and track_concat_channels_local > 0
                and patch_w.shape[1] >= track_concat_channels_local
            ):
                old_in = int(patch_w.shape[1] - track_concat_channels_local)

                def _mask_pretrained_patch_channels(grad, old_in_channels=old_in):
                    if grad is None:
                        return grad
                    masked = grad.clone()
                    if old_in_channels > 0:
                        masked[:, :old_in_channels, ...] = 0
                    return masked

                patch_new_grad_mask_handle = patch_w.register_hook(_mask_pretrained_patch_channels)
        new_only_phase_active = new_only
        if accelerator.is_main_process:
            active_now = sum(
                p.numel()
                for name, p in unwrapped_model.named_parameters()
                if (name in trainable_name_set) and p.requires_grad
            )
            phase_name = "new_only" if new_only else "full_trainable"
            msg = f"Trainability phase={phase_name}, active_params={active_now}"
            logger.info(msg)
            accelerator.print(msg)

    if new_params_only_steps_track > 0 and global_step < new_params_only_steps_track:
        _set_trainability_phase(new_only=True)
    else:
        _set_trainability_phase(new_only=False)

    def _run_validation_once() -> Optional[Dict[str, float]]:
        nonlocal text_encoder_on_gpu
        if val_dataloader is None or args.validation_steps_track <= 0:
            return None
        max_val_batches = max(1, int(args.validation_max_batches_track))
        model_unwrapped = accelerator.unwrap_model(transformer3d_track)
        was_training = bool(model_unwrapped.training)
        model_unwrapped.eval()
        val_loss_sum = 0.0
        val_batches = 0
        moved_text_encoder_for_val = False
        try:
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    if val_batch_idx >= max_val_batches:
                        break
                    if args.input_mode_track == "latent":
                        val_latents = val_batch["latents"].to(accelerator.device, dtype=weight_dtype)
                    else:
                        val_pixel_values = val_batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                        val_pixel_values = val_pixel_values.permute(0, 2, 1, 3, 4).contiguous()
                        val_latents = vae.encode(val_pixel_values)[0].sample()

                    if args.input_mode_track == "latent" and ("precomputed_prompt_embeds" in val_batch):
                        val_prompt_embeds = val_batch["precomputed_prompt_embeds"].to(
                            accelerator.device, dtype=weight_dtype
                        )
                    else:
                        if not text_encoder_on_gpu:
                            text_encoder.to(accelerator.device, dtype=weight_dtype)
                            text_encoder_on_gpu = True
                            moved_text_encoder_for_val = True
                        val_prompt_ids = tokenizer(
                            val_batch["text"],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        )
                        val_prompt_embeds = text_encoder(
                            val_prompt_ids.input_ids.to(accelerator.device),
                            attention_mask=val_prompt_ids.attention_mask.to(accelerator.device),
                        )[0]

                    val_bsz, _, val_num_frames, val_height, val_width = val_latents.shape
                    val_noise = torch.randn_like(val_latents)
                    val_indices = idx_sampling(val_bsz, generator=None, device=val_latents.device).long().cpu()
                    val_timesteps = noise_scheduler.timesteps[val_indices].to(device=val_latents.device)
                    val_sigmas = noise_scheduler.sigmas.to(device=val_latents.device, dtype=val_latents.dtype)
                    val_schedule_timesteps = noise_scheduler.timesteps.to(val_latents.device)
                    val_step_indices = [(val_schedule_timesteps == t).nonzero().item() for t in val_timesteps]
                    val_sigma = val_sigmas[val_step_indices].view(-1, 1, 1, 1, 1)
                    val_noisy_latents = (1.0 - val_sigma) * val_latents + val_sigma * val_noise
                    val_target = val_noise - val_latents

                    val_target_shape = (vae.latent_channels, val_num_frames, val_width, val_height)
                    val_seq_len = math.ceil(
                        (val_target_shape[2] * val_target_shape[3])
                        / (
                            accelerator.unwrap_model(transformer3d_track).config.patch_size[1]
                            * accelerator.unwrap_model(transformer3d_track).config.patch_size[2]
                        )
                        * val_target_shape[1]
                    )
                    val_track_condition = val_batch.get("track_condition", None) if args.use_track_condition else None
                    val_clip_fea = val_batch.get("precomputed_clip_feature", None)
                    if val_clip_fea is not None:
                        val_clip_fea = val_clip_fea.to(accelerator.device, dtype=weight_dtype)
                    val_y = None
                    if (
                        args.input_mode_track == "latent"
                        and args.train_mode != "normal"
                        and args.use_first_frame_condition_track
                    ):
                        if "first_frame_vae_latent" in val_batch:
                            val_masked_video_latents = val_batch["first_frame_vae_latent"].to(
                                accelerator.device,
                                dtype=weight_dtype,
                            )
                            if val_masked_video_latents.shape[2:] != val_latents.shape[2:]:
                                val_masked_video_latents = F.interpolate(
                                    val_masked_video_latents,
                                    size=val_latents.shape[2:],
                                    mode="trilinear",
                                    align_corners=False,
                                )
                            val_mask_latents = torch.zeros(
                                (
                                    val_masked_video_latents.shape[0],
                                    4,
                                    val_masked_video_latents.shape[2],
                                    val_masked_video_latents.shape[3],
                                    val_masked_video_latents.shape[4],
                                ),
                                device=accelerator.device,
                                dtype=weight_dtype,
                            )
                            val_mask_latents[:, :, 0:1, :, :] = 1.0
                        else:
                            val_masked_video_latents = torch.zeros_like(val_latents)
                            val_mask_latents = torch.zeros(
                                (
                                    val_latents.shape[0],
                                    4,
                                    val_latents.shape[2],
                                    val_latents.shape[3],
                                    val_latents.shape[4],
                                ),
                                device=accelerator.device,
                                dtype=weight_dtype,
                            )
                        val_y = torch.cat([val_mask_latents, val_masked_video_latents], dim=1)
                        if args.track_condition_mode == "wan_move":
                            val_y_track = val_y[:, -val_latents.shape[1] :]
                            val_y_track, _ = _apply_wan_move_feature_replace(
                                val_y_track,
                                val_track_condition,
                                temporal_stride=wan_move_temporal_stride,
                            )
                            val_y = val_y.clone()
                            val_y[:, -val_latents.shape[1] :] = val_y_track

                    val_noise_pred = _forward_transformer_with_track_mode(
                        transformer3d_track,
                        args.track_condition_mode,
                        val_track_condition,
                        x=val_noisy_latents,
                        context=val_prompt_embeds,
                        t=val_timesteps,
                        seq_len=val_seq_len,
                        y=val_y,
                        clip_fea=val_clip_fea,
                    )
                    val_loss = F.mse_loss(val_noise_pred.float(), val_target.float(), reduction="mean")
                    val_loss_gathered = accelerator.gather_for_metrics(val_loss.detach().float().view(1))
                    val_loss_sum += float(val_loss_gathered.mean().item())
                    val_batches += 1
        finally:
            if moved_text_encoder_for_val and is_latent_mode:
                text_encoder.to("cpu")
                text_encoder_on_gpu = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if was_training:
                model_unwrapped.train()

        if val_batches == 0:
            return None
        return {
            "val_loss_track": val_loss_sum / float(val_batches),
            "val_batches_track": float(val_batches),
        }

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    first_step_peak_memory_logged = False
    first_step_peak_memory_started = False
    verify_checked_batches = 0
    verify_failed_batches = 0
    verify_max_batches = int(args.verify_first_frame_vae_latent_max_batches)
    if args.verify_first_frame_vae_latent_track and verify_max_batches <= 0:
        verify_max_batches = 1
    debug_baseline = None
    track_health_prev_patch_added = None
    track_health_prev_track_head: Dict[str, torch.Tensor] = {}
    unwrapped_for_health = accelerator.unwrap_model(transformer3d_track)
    track_concat_channels = int(getattr(unwrapped_for_health, "track_concat_channels", 0))
    with torch.no_grad():
        patch_w_init = getattr(getattr(unwrapped_for_health, "patch_embedding", None), "weight", None)
        if (
            patch_w_init is not None
            and track_concat_channels > 0
            and patch_w_init.shape[1] >= track_concat_channels
        ):
            track_health_prev_patch_added = (
                patch_w_init.detach().float()[:, -track_concat_channels:, ...].cpu().clone()
            )
        for name, p in unwrapped_for_health.named_parameters():
            if p.requires_grad and ("track_head" in name):
                track_health_prev_track_head[name] = p.detach().float().cpu().clone()
    if args.debug_weight_update_track:
        # Baseline snapshot for measuring per-step parameter update magnitudes.
        debug_baseline = {
            name: p.detach().float().clone()
            for name, p in trainable_named_params
        }
    for _ in range(args.num_train_epochs):
        for batch in train_dataloader:
            if (
                new_params_only_steps_track > 0
                and new_only_phase_active
                and global_step >= new_params_only_steps_track
            ):
                _set_trainability_phase(new_only=False)
            with accelerator.accumulate(transformer3d_track):
                track_health_metrics = {}
                if (
                    (not first_step_peak_memory_started)
                    and torch.cuda.is_available()
                    and accelerator.device.type == "cuda"
                ):
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                    first_step_peak_memory_started = True
                if (
                    args.debug_memory_track
                    and torch.cuda.is_available()
                    and accelerator.device.type == "cuda"
                ):
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                if args.input_mode_track == "latent":
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
                    with torch.no_grad():
                        latents = vae.encode(pixel_values)[0].sample()

                with torch.no_grad():
                    if args.input_mode_track == "latent" and ("precomputed_prompt_embeds" in batch):
                        prompt_embeds = batch["precomputed_prompt_embeds"].to(
                            accelerator.device, dtype=weight_dtype
                        )
                    else:
                        if not text_encoder_on_gpu:
                            text_encoder.to(accelerator.device, dtype=weight_dtype)
                            text_encoder_on_gpu = True
                        prompt_ids = tokenizer(
                            batch["text"],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        )
                        prompt_embeds = text_encoder(
                            prompt_ids.input_ids.to(accelerator.device),
                            attention_mask=prompt_ids.attention_mask.to(accelerator.device),
                        )[0]
                        if is_latent_mode:
                            text_encoder.to("cpu")
                            text_encoder_on_gpu = False
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                bsz, _, num_frames, height, width = latents.shape
                noise = torch.randn_like(latents)
                indices = idx_sampling(bsz, generator=None, device=latents.device).long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
                schedule_timesteps = noise_scheduler.timesteps.to(latents.device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                sigma = sigmas[step_indices].view(-1, 1, 1, 1, 1)
                noisy_latents = (1.0 - sigma) * latents + sigma * noise
                target = noise - latents

                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3])
                    / (
                        accelerator.unwrap_model(transformer3d_track).config.patch_size[1]
                        * accelerator.unwrap_model(transformer3d_track).config.patch_size[2]
                    )
                    * target_shape[1]
                )

                track_condition = batch.get("track_condition", None) if args.use_track_condition else None
                track_condition_stats = _summarize_track_condition(track_condition)
                clip_fea = batch.get("precomputed_clip_feature", None)
                if clip_fea is not None:
                    clip_fea = clip_fea.to(accelerator.device, dtype=weight_dtype)
                y = None
                first_frame_drop_ratio = 0.0
                wan_move_condition_stats = _empty_wan_move_condition_stats(
                    enabled=args.track_condition_mode == "wan_move"
                )
                if (
                    args.input_mode_track == "latent"
                    and args.train_mode != "normal"
                    and args.use_first_frame_condition_track
                    and ("first_frame_vae_latent" in batch)
                ):
                    masked_video_latents = batch["first_frame_vae_latent"].to(
                        accelerator.device,
                        dtype=weight_dtype,
                    )
                    if masked_video_latents.shape[2:] != latents.shape[2:]:
                        masked_video_latents = F.interpolate(
                            masked_video_latents,
                            size=latents.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                    if (
                        args.verify_first_frame_vae_latent_track
                        and verify_checked_batches < verify_max_batches
                        and ("first_frame_pixel_values" in batch)
                    ):
                        assert image_processor is not None and mask_processor is not None
                        first_frame_verify = batch["first_frame_pixel_values"].to(
                            accelerator.device, dtype=torch.float32
                        )
                        bsz_v, _, ff_h_v, ff_w_v = first_frame_verify.shape
                        video_length_v = int(args.video_sample_n_frames)

                        video_v = torch.tile(first_frame_verify.unsqueeze(2), [1, 1, video_length_v, 1, 1])
                        mask_video_v = torch.zeros(
                            (bsz_v, 1, video_length_v, ff_h_v, ff_w_v),
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                        mask_video_v[:, :, 1:] = 255.0

                        init_video_v = image_processor.preprocess(
                            rearrange(video_v, "b c f h w -> (b f) c h w"),
                            height=ff_h_v,
                            width=ff_w_v,
                        )
                        init_video_v = init_video_v.to(dtype=torch.float32, device=accelerator.device)
                        init_video_v = rearrange(init_video_v, "(b f) c h w -> b c f h w", f=video_length_v)

                        mask_condition_v = mask_processor.preprocess(
                            rearrange(mask_video_v, "b c f h w -> (b f) c h w"),
                            height=ff_h_v,
                            width=ff_w_v,
                        )
                        mask_condition_v = mask_condition_v.to(
                            dtype=torch.float32, device=accelerator.device
                        )
                        mask_condition_v = rearrange(
                            mask_condition_v, "(b f) c h w -> b c f h w", f=video_length_v
                        )
                        masked_video_v = init_video_v * (torch.tile(mask_condition_v, [1, 3, 1, 1, 1]) < 0.5)
                        with torch.no_grad():
                            masked_video_latents_v = _encode_masked_video_like_pipeline(
                                masked_video=masked_video_v,
                                vae=vae,
                                device=accelerator.device,
                            ).to(dtype=weight_dtype)
                        if masked_video_latents_v.shape[2:] != latents.shape[2:]:
                            masked_video_latents_v = F.interpolate(
                                masked_video_latents_v,
                                size=latents.shape[2:],
                                mode="trilinear",
                                align_corners=False,
                            )

                        diff = (masked_video_latents.detach().float() - masked_video_latents_v.detach().float()).abs()
                        diff_max = float(diff.max().item())
                        diff_mean = float(diff.mean().item())
                        verify_checked_batches += 1
                        if diff_max > float(args.verify_first_frame_vae_latent_tol):
                            verify_failed_batches += 1
                            accelerator.print(
                                "[verify-first-frame-vae] "
                                f"batch={verify_checked_batches} diff_max={diff_max:.6e} "
                                f"diff_mean={diff_mean:.6e} tol={args.verify_first_frame_vae_latent_tol:.6e} "
                                "status=FAIL"
                            )
                        else:
                            accelerator.print(
                                "[verify-first-frame-vae] "
                                f"batch={verify_checked_batches} diff_max={diff_max:.6e} "
                                f"diff_mean={diff_mean:.6e} tol={args.verify_first_frame_vae_latent_tol:.6e} "
                                "status=OK"
                            )
                    bsz_ff, _, latent_f, latent_h, latent_w = masked_video_latents.shape
                    mask_latents = torch.zeros(
                        (bsz_ff, 4, latent_f, latent_h, latent_w),
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    mask_latents[:, :, 0:1, :, :] = 1.0
                    y = torch.cat([mask_latents, masked_video_latents], dim=1)
                elif (
                    args.input_mode_track == "latent"
                    and args.train_mode != "normal"
                    and args.use_first_frame_condition_track
                    and ("first_frame_pixel_values" in batch)
                ):
                    assert image_processor is not None and mask_processor is not None
                    first_frame = batch["first_frame_pixel_values"].to(
                        accelerator.device, dtype=torch.float32
                    )  # [B,3,H,W] in [0,1]
                    bsz_ff, _, ff_h, ff_w = first_frame.shape
                    video_length = int(args.video_sample_n_frames)

                    # Match get_image_to_video_latent(single first frame) + pipeline preprocess path.
                    video = torch.tile(first_frame.unsqueeze(2), [1, 1, video_length, 1, 1])
                    mask_video = torch.zeros(
                        (bsz_ff, 1, video_length, ff_h, ff_w),
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    mask_video[:, :, 1:] = 255.0

                    init_video = image_processor.preprocess(
                        rearrange(video, "b c f h w -> (b f) c h w"),
                        height=ff_h,
                        width=ff_w,
                    )
                    init_video = init_video.to(dtype=torch.float32, device=accelerator.device)
                    init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)

                    mask_condition = mask_processor.preprocess(
                        rearrange(mask_video, "b c f h w -> (b f) c h w"),
                        height=ff_h,
                        width=ff_w,
                    )
                    mask_condition = mask_condition.to(dtype=torch.float32, device=accelerator.device)
                    mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)

                    masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)
                    with torch.no_grad():
                        masked_video_latents = _encode_masked_video_like_pipeline(
                            masked_video=masked_video,
                            vae=vae,
                            device=accelerator.device,
                        ).to(dtype=weight_dtype)

                    mask_condition = torch.concat(
                        [
                            torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
                            mask_condition[:, :, 1:],
                        ],
                        dim=2,
                    )
                    mask_condition = mask_condition.view(
                        bsz_ff, mask_condition.shape[2] // 4, 4, ff_h, ff_w
                    ).transpose(1, 2)
                    mask_latents = _resize_mask_like_pipeline(
                        1.0 - mask_condition,
                        masked_video_latents,
                        process_first_frame_only=True,
                    ).to(device=accelerator.device, dtype=weight_dtype)

                    # Safety guard for any shape drift.
                    if masked_video_latents.shape[2:] != latents.shape[2:]:
                        masked_video_latents = F.interpolate(
                            masked_video_latents,
                            size=latents.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                    if mask_latents.shape[2:] != latents.shape[2:]:
                        mask_latents = F.interpolate(
                            mask_latents,
                            size=latents.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                    y = torch.cat([mask_latents, masked_video_latents], dim=1)
                elif (
                    args.input_mode_track == "latent"
                    and args.train_mode != "normal"
                    and args.use_first_frame_condition_track
                ):
                    # First-frame conditioning requested but sample has no first_frame png.
                    # Fallback to zeros to keep batch shape consistent.
                    mask_latents = torch.zeros(
                        (latents.shape[0], 4, latents.shape[2], latents.shape[3], latents.shape[4]),
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    masked_video_latents = torch.zeros_like(latents)
                    y = torch.cat([mask_latents, masked_video_latents], dim=1)

                if y is not None and args.track_condition_mode == "wan_move":
                    y_track = y[:, -latents.shape[1] :]
                    y_track, wan_move_condition_stats = _apply_wan_move_feature_replace(
                        y_track,
                        track_condition,
                        temporal_stride=wan_move_temporal_stride,
                    )
                    y = y.clone()
                    y[:, -latents.shape[1] :] = y_track

                if (
                    y is not None
                    and args.input_mode_track == "latent"
                    and args.train_mode != "normal"
                    and args.use_first_frame_condition_track
                ):
                    ff_drop_prob = float(args.first_frame_condition_drop_prob_track)
                    if ff_drop_prob > 0.0:
                        if ff_drop_prob >= 1.0:
                            first_frame_drop_mask = torch.ones(
                                (y.shape[0],),
                                dtype=torch.bool,
                                device=y.device,
                            )
                        else:
                            first_frame_drop_mask = (
                                torch.rand((y.shape[0],), device=y.device) < ff_drop_prob
                            )
                        first_frame_drop_ratio = float(first_frame_drop_mask.float().mean().item())
                        if torch.any(first_frame_drop_mask):
                            y = y.clone()
                            y[first_frame_drop_mask] = 0
                            if clip_fea is not None and clip_fea.shape[0] == y.shape[0]:
                                clip_fea = clip_fea.clone()
                                clip_fea[first_frame_drop_mask] = 0

                first_frame_condition_stats = {
                    "first_frame_stats/condition_enabled": float(
                        args.input_mode_track == "latent"
                        and args.train_mode != "normal"
                        and args.use_first_frame_condition_track
                    ),
                    "first_frame_stats/drop_prob": float(args.first_frame_condition_drop_prob_track),
                    "first_frame_stats/dropped_sample_ratio": float(first_frame_drop_ratio),
                    "first_frame_stats/y_is_none": float(y is None),
                }
                noise_pred = _forward_transformer_with_track_mode(
                    transformer3d_track,
                    args.track_condition_mode,
                    track_condition,
                    x=noisy_latents,
                    context=prompt_embeds,
                    t=timesteps,
                    seq_len=seq_len,
                    y=y,
                    clip_fea=clip_fea,
                )
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(transformer3d_track)
                    grad_param_total = 0
                    grad_params_with_grad = 0
                    grad_params_nonzero = 0
                    grad_global_norm_sq = 0.0
                    track_head_grad_norm_sq = 0.0
                    track_head_grad_abs_mean_sum = 0.0
                    track_head_grad_param_count = 0
                    for name, p in unwrapped_model.named_parameters():
                        if not p.requires_grad:
                            continue
                        grad_param_total += 1
                        g = p.grad
                        if g is None:
                            continue
                        grad_params_with_grad += 1
                        g_f = g.detach().float()
                        g_norm = float(g_f.norm().item())
                        if g_norm > 0.0:
                            grad_params_nonzero += 1
                        grad_global_norm_sq += g_norm * g_norm
                        if "track_head" in name:
                            track_head_grad_norm_sq += g_norm * g_norm
                            track_head_grad_abs_mean_sum += float(g_f.abs().mean().item())
                            track_head_grad_param_count += 1

                    patch_added_weight_grad_norm = 0.0
                    patch_added_grad_abs_mean = 0.0
                    patch_w = getattr(getattr(unwrapped_model, "patch_embedding", None), "weight", None)
                    if (
                        patch_w is not None
                        and patch_w.grad is not None
                        and track_concat_channels > 0
                        and patch_w.shape[1] >= track_concat_channels
                    ):
                        g_added = patch_w.grad.detach().float()[:, -track_concat_channels:, ...]
                        patch_added_weight_grad_norm = float(g_added.norm().item())
                        patch_added_grad_abs_mean = float(g_added.abs().mean().item())

                    track_health_metrics.update(
                        {
                            "track_health/grad_param_total": float(grad_param_total),
                            "track_health/grad_with_grad_ratio": (
                                float(grad_params_with_grad) / float(max(grad_param_total, 1))
                            ),
                            "track_health/grad_nonzero_ratio": (
                                float(grad_params_nonzero) / float(max(grad_param_total, 1))
                            ),
                            "track_health/grad_global_norm": float(math.sqrt(max(grad_global_norm_sq, 0.0))),
                            "track_health/track_head_grad_norm": float(
                                math.sqrt(max(track_head_grad_norm_sq, 0.0))
                            ),
                            "track_health/track_head_grad_abs_mean_avg": (
                                track_head_grad_abs_mean_sum / float(max(track_head_grad_param_count, 1))
                            ),
                            "patch_added_weight_grad_norm": patch_added_weight_grad_norm,
                            "track_health/patch_added_grad_norm": patch_added_weight_grad_norm,
                            "track_health/patch_added_weight_grad_norm": patch_added_weight_grad_norm,
                            "track_health/patch_added_grad_abs_mean": patch_added_grad_abs_mean,
                        }
                    )

                if args.debug_weight_update_track and accelerator.is_main_process:
                    grads_none = 0
                    grads_zero = 0
                    grads_nonzero = 0
                    track_head_grad_abs_mean = 0.0
                    track_head_grad_norm_sum = 0.0
                    track_head_grad_param_count = 0
                    for _, p in trainable_named_params:
                        g = p.grad
                        if g is None:
                            grads_none += 1
                            continue
                        if torch.count_nonzero(g).item() == 0:
                            grads_zero += 1
                        else:
                            grads_nonzero += 1
                    for name, p in trainable_named_params:
                        if "track_head" not in name or p.grad is None:
                            continue
                        g = p.grad.detach().float()
                        track_head_grad_abs_mean += g.abs().mean().item()
                        track_head_grad_norm_sum += g.norm().item()
                        track_head_grad_param_count += 1
                    accelerator.print(
                        f"[weight-debug] grads: none={grads_none} zero={grads_zero} nonzero={grads_nonzero}"
                    )

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(transformer3d_track)
                    track_head_update_norm_sum = 0.0
                    track_head_rel_update_sum = 0.0
                    track_head_update_param_count = 0
                    track_head_updated_nonzero = 0
                    with torch.no_grad():
                        for name, p in unwrapped_model.named_parameters():
                            if (not p.requires_grad) or ("track_head" not in name):
                                continue
                            prev = track_health_prev_track_head.get(name, None)
                            current = p.detach().float()
                            if prev is None:
                                track_health_prev_track_head[name] = current.cpu().clone()
                                continue
                            prev_dev = prev.to(current.device)
                            delta = current - prev_dev
                            update_norm = float(delta.norm().item())
                            param_norm = float(current.norm().item())
                            rel_update = update_norm / (param_norm + 1e-12)
                            track_head_update_norm_sum += update_norm
                            track_head_rel_update_sum += rel_update
                            track_head_update_param_count += 1
                            if update_norm > 0.0:
                                track_head_updated_nonzero += 1
                            track_health_prev_track_head[name] = current.cpu().clone()

                        patch_added_update_norm = 0.0
                        patch_added_rel_update = 0.0
                        patch_added_weight_norm = 0.0
                        patch_w = getattr(getattr(unwrapped_model, "patch_embedding", None), "weight", None)
                        if (
                            patch_w is not None
                            and track_concat_channels > 0
                            and patch_w.shape[1] >= track_concat_channels
                        ):
                            current_added = patch_w.detach().float()[:, -track_concat_channels:, ...]
                            patch_added_weight_norm = float(current_added.norm().item())
                            if track_health_prev_patch_added is not None:
                                prev_added = track_health_prev_patch_added.to(current_added.device)
                                delta_added = current_added - prev_added
                                patch_added_update_norm = float(delta_added.norm().item())
                                patch_added_rel_update = patch_added_update_norm / (
                                    patch_added_weight_norm + 1e-12
                                )
                            track_health_prev_patch_added = current_added.cpu().clone()

                    track_health_metrics.update(
                        {
                            "track_health/track_head_update_norm_sum": track_head_update_norm_sum,
                            "track_health/track_head_rel_update_avg": (
                                track_head_rel_update_sum / float(max(track_head_update_param_count, 1))
                            ),
                            "track_health/track_head_updated_nonzero_ratio": (
                                float(track_head_updated_nonzero)
                                / float(max(track_head_update_param_count, 1))
                            ),
                            "track_health/patch_added_weight_norm": patch_added_weight_norm,
                            "track_health/patch_added_update_norm": patch_added_update_norm,
                            "track_health/patch_added_rel_update": patch_added_rel_update,
                        }
                    )

                if (
                    args.debug_weight_update_track
                    and accelerator.is_main_process
                    and debug_baseline is not None
                ):
                    update_stats = []
                    updated_nonzero = 0
                    track_head_update_norm_sum = 0.0
                    track_head_rel_update_sum = 0.0
                    track_head_update_param_count = 0
                    for name, p in trainable_named_params:
                        current = p.detach().float()
                        prev = debug_baseline[name].to(current.device)
                        delta = current - prev
                        update_norm = delta.norm().item()
                        param_norm = current.norm().item()
                        rel_update = update_norm / (param_norm + 1e-12)
                        update_stats.append((name, update_norm, rel_update))
                        if update_norm > 0:
                            updated_nonzero += 1
                        if "track_head" in name:
                            track_head_update_norm_sum += update_norm
                            track_head_rel_update_sum += rel_update
                            track_head_update_param_count += 1
                        debug_baseline[name] = current.cpu().clone()

                    update_stats.sort(key=lambda x: x[1], reverse=True)
                    topk = max(1, int(args.debug_weight_update_topk_track))
                    accelerator.print(
                        f"[weight-debug] updated params(nonzero delta): {updated_nonzero}/{len(update_stats)}"
                    )
                    for name, up_norm, rel_up in update_stats[:topk]:
                        accelerator.print(
                            f"[weight-debug] {name}: update_norm={up_norm:.6e}, rel_update={rel_up:.6e}"
                        )
                    if track_head_update_param_count > 0:
                        accelerator.print(
                            "[weight-debug] track_head summary: "
                            f"grad_abs_mean_avg={track_head_grad_abs_mean / max(track_head_grad_param_count, 1):.6e}, "
                            f"grad_norm_sum={track_head_grad_norm_sum:.6e}, "
                            f"update_norm_sum={track_head_update_norm_sum:.6e}, "
                            f"rel_update_avg={track_head_rel_update_sum / track_head_update_param_count:.6e}"
                        )

                if (
                    args.debug_memory_track
                    and torch.cuda.is_available()
                    and accelerator.device.type == "cuda"
                    and accelerator.is_main_process
                ):
                    current_alloc = torch.cuda.memory_allocated(accelerator.device) / (1024 ** 3)
                    current_reserved = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3)
                    peak_alloc = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
                    peak_reserved = torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3)
                    mem_msg = (
                        f"CUDA memory step={global_step + int(accelerator.sync_gradients)} "
                        f"alloc={current_alloc:.2f}GB reserved={current_reserved:.2f}GB "
                        f"peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB"
                    )
                    logger.info(mem_msg)
                    accelerator.print(mem_msg)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if (
                        accelerator.is_main_process
                        and int(args.track_debug_vis_steps) > 0
                        and args.track_condition_mode == "track_head"
                        and (global_step % int(args.track_debug_vis_steps) == 0)
                    ):
                        try:
                            vis_root = _export_track_debug_visualization(
                                args=args,
                                accelerator=accelerator,
                                vae=vae,
                                transformer3d_track=transformer3d_track,
                                latents=latents,
                                track_condition=track_condition,
                                weight_dtype=weight_dtype,
                                global_step=global_step,
                            )
                            if vis_root is not None:
                                logger.info(
                                    "Saved track-debug visualization at step=%d to %s",
                                    global_step,
                                    vis_root,
                                )
                        except Exception:
                            logger.exception(
                                "Failed to export track-debug visualization at step=%d",
                                global_step,
                            )
                    lr_values = lr_scheduler.get_last_lr()
                    log_payload = {
                        "train_loss_track": loss.detach().item(),
                        "lr_track": lr_values[0],
                    }
                    if len(lr_values) > 1:
                        log_payload["lr_track_new_layers"] = lr_values[0]
                        log_payload["lr_track_base_blocks"] = lr_values[1]
                    log_payload.update(track_condition_stats)
                    log_payload.update(first_frame_condition_stats)
                    log_payload.update(wan_move_condition_stats)
                    model_track_debug = getattr(
                        accelerator.unwrap_model(transformer3d_track),
                        "_last_track_debug",
                        None,
                    )
                    if isinstance(model_track_debug, dict):
                        log_payload.update(
                            {
                                f"track_model/{k}": float(v)
                                for k, v in model_track_debug.items()
                            }
                        )
                        for key in (
                            "patch_track_out_norm",
                            "patch_existing_out_norm",
                            "patch_track_to_existing_out_ratio",
                            "track_head_out_std",
                        ):
                            if key in model_track_debug:
                                log_payload[key] = float(model_track_debug[key])
                    log_payload.update(track_health_metrics)
                    if args.debug_weight_update_track and accelerator.is_main_process:
                        log_payload.update(
                            {
                            "track_debug/grads_none": grads_none,
                            "track_debug/grads_zero": grads_zero,
                            "track_debug/grads_nonzero": grads_nonzero,
                            "track_debug/updated_nonzero_params": updated_nonzero,
                        }
                    )
                    if (
                        val_dataloader is not None
                        and args.validation_steps_track > 0
                        and global_step % args.validation_steps_track == 0
                    ):
                        if accelerator.is_main_process:
                            logger.info(
                                "Running validation at step=%d (max_batches=%d)",
                                global_step,
                                int(args.validation_max_batches_track),
                            )
                        val_metrics = _run_validation_once()
                        if val_metrics is not None:
                            log_payload.update(val_metrics)
                            if accelerator.is_main_process:
                                logger.info(
                                    "Validation done at step=%d: val_loss_track=%.6f, val_batches_track=%d",
                                    global_step,
                                    float(val_metrics["val_loss_track"]),
                                    int(val_metrics["val_batches_track"]),
                                )
                    accelerator.log(log_payload, step=global_step)

                if (
                    (not first_step_peak_memory_logged)
                    and torch.cuda.is_available()
                    and accelerator.device.type == "cuda"
                    and accelerator.is_main_process
                ):
                    first_step_peak_alloc_gb = (
                        torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
                    )
                    mem_first_msg = (
                        "First optimizer step peak CUDA memory allocated="
                        f"{first_step_peak_alloc_gb:.2f}GB"
                    )
                    logger.info(mem_first_msg)
                    accelerator.print(mem_first_msg)
                    first_step_peak_memory_logged = True

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        _write_trainer_state_track(save_path, global_step)
                    logger.info("Saved state to %s", save_path)

                if args.dry_run_track:
                    logger.info("dry_run_track enabled: stopping after first optimizer step.")
                    break
                if global_step >= args.max_train_steps:
                    break

        if args.dry_run_track or global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_path = os.path.join(ckpt_root, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if accelerator.is_main_process:
        if args.verify_first_frame_vae_latent_track:
            accelerator.print(
                "[verify-first-frame-vae] "
                f"checked_batches={verify_checked_batches} failed_batches={verify_failed_batches}"
            )
        _write_trainer_state_track(save_path, global_step)
        logger.info("Saved final state to %s", save_path)
    accelerator.end_training()


if __name__ == "__main__":
    main()
