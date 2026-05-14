import argparse
import csv
import json
import os
import random
import sys
import tempfile
import types
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from PIL import Image, ImageDraw

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
    WanTransformer3DModel,
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


def _tracks_look_normalized(tracks: np.ndarray) -> bool:
    """Match WanTransformer3DModelTrack._map_to_latent_grid heuristic."""
    t = np.asarray(tracks, dtype=np.float32)
    mx = float(np.nanmax(t))
    mn = float(np.nanmin(t))
    return mx <= 2.0 and mn >= -0.5


def _str2bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _select_track_indices_uniform(num_points: int, max_points: int) -> np.ndarray:
    """Deterministic point sampling that preserves full spatial coverage."""
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    idx = np.linspace(0, num_points - 1, max_points, dtype=np.float64)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, num_points - 1)
    idx = np.unique(idx)

    if idx.size < max_points:
        need = max_points - idx.size
        pool = np.setdiff1d(np.arange(num_points, dtype=np.int64), idx)
        if pool.size > 0:
            fill_pos = np.linspace(0, pool.size - 1, need, dtype=np.float64)
            fill = pool[np.round(fill_pos).astype(np.int64)]
            idx = np.sort(np.concatenate([idx, fill]))

    if idx.size > max_points:
        idx = idx[:max_points]
    return idx.astype(np.int64, copy=False)


def _select_track_indices_random(
    num_points: int,
    max_points: int,
    *,
    sort_selected_indices: bool,
    seed: Optional[int],
) -> np.ndarray:
    """Random track point sampling with optional index-order preservation."""
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_points)[:max_points]
    if sort_selected_indices:
        idx = np.sort(idx)
    return idx.astype(np.int64, copy=False)


def _select_track_indices(
    num_points: int,
    max_points: int,
    *,
    sample_mode: str,
    sort_selected_indices: bool,
    seed: Optional[int],
) -> np.ndarray:
    mode = str(sample_mode).strip().lower()
    if mode == "uniform":
        return _select_track_indices_uniform(num_points=num_points, max_points=max_points)
    if mode == "random":
        return _select_track_indices_random(
            num_points=num_points,
            max_points=max_points,
            sort_selected_indices=sort_selected_indices,
            seed=seed,
        )
    raise ValueError(f"Unsupported track point sample mode: {sample_mode}")


def _track_inbound_ratio(
    tracks: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    frame_slice: Optional[slice] = None,
) -> float:
    t = np.asarray(tracks, dtype=np.float32)
    v = np.asarray(visibility, dtype=np.float32)
    if t.ndim != 3 or t.shape[-1] != 2:
        return 0.0
    if v.ndim == 3 and v.shape[-1] == 1:
        v = v.squeeze(-1)
    if v.ndim != 2:
        v = np.ones(t.shape[:2], dtype=np.float32)

    if frame_slice is not None:
        t = t[frame_slice]
        v = v[frame_slice]

    if t.size == 0:
        return 1.0

    if v.shape[:2] != t.shape[:2]:
        v = np.ones(t.shape[:2], dtype=np.float32)

    valid = v > 0.5
    if not np.any(valid):
        valid = np.ones(t.shape[:2], dtype=bool)

    x = t[..., 0][valid]
    y = t[..., 1][valid]
    if x.size == 0:
        return 1.0

    in_bound = (
        (x >= -1.0)
        & (x <= float(width) + 1.0)
        & (y >= -1.0)
        & (y <= float(height) + 1.0)
    )
    return float(in_bound.mean())


def _overlay_draw_line(
    rgb: Image.Image,
    coord_a: Tuple[int, int],
    coord_b: Tuple[int, int],
    color: Tuple[int, int, int],
    linewidth: int,
) -> Image.Image:
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_a[0], coord_a[1], coord_b[0], coord_b[1]),
        fill=color,
        width=max(1, int(linewidth)),
    )
    return rgb


def _overlay_draw_circle(
    rgb: Image.Image,
    coord: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    visible: bool,
) -> Image.Image:
    draw = ImageDraw.Draw(rgb)
    x, y = int(coord[0]), int(coord[1])
    r = int(radius)
    box = (x - r, y - r, x + r, y + r)
    col = color
    draw.ellipse(box, fill=col if visible else None, outline=col)
    return rgb


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


def _load_track_condition(
    track_file_path: str,
    normalize: bool,
    normalize_height: int,
    normalize_width: int,
    track_max_points: Optional[int],
    track_point_sample_mode: str,
    track_sort_selected_indices: bool,
    track_point_sample_seed: Optional[int],
    track_point_id_mode: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:

    if os.environ.get("PDB_DEBUG", "0") == "1":
        import pdb; pdb.set_trace()

    data = np.load(track_file_path)

    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        raise KeyError(f"Track file missing tracks key: {track_file_path}")

    if "visibility_compressed" in data:
        visibility = data["visibility_compressed"]
    elif "visibility" in data:
        visibility = data["visibility"]
    else:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)

    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_file_path}")
    if visibility.ndim != 2:
        raise ValueError(f"Unexpected visibility shape {visibility.shape} in {track_file_path}")

    num_points_before = int(tracks.shape[1])
    keep_idx = np.arange(num_points_before, dtype=np.int64)
    track_point_id_mode = str(track_point_id_mode).strip().lower()
    if track_point_id_mode not in {"original", "local"}:
        raise ValueError(f"Unsupported track_point_id_mode: {track_point_id_mode}")
    if track_max_points is not None and int(track_max_points) > 0:
        track_max_points = int(track_max_points)
        keep_idx = _select_track_indices(
            num_points=num_points_before,
            max_points=track_max_points,
            sample_mode=track_point_sample_mode,
            sort_selected_indices=track_sort_selected_indices,
            seed=track_point_sample_seed,
        )
        tracks = tracks[:, keep_idx]
        visibility = visibility[:, keep_idx]
        seed_str = "n/a" if track_point_sample_mode != "random" else str(track_point_sample_seed)
        print(
            "[info] track_max_points applied: "
            f"mode={track_point_sample_mode} "
            f"sort_selected_indices={track_sort_selected_indices} "
            f"seed={seed_str} "
            f"point_id_mode={track_point_id_mode} "
            f"kept={tracks.shape[1]}/{num_points_before}"
        )

    tracks_t = torch.as_tensor(tracks, dtype=torch.float32, device=device)
    visibility_t = torch.as_tensor(visibility, dtype=torch.float32, device=device)
    point_mask_t = torch.ones(
        (tracks_t.shape[1],),
        dtype=torch.bool,
        device=device,
    )
    if track_point_id_mode == "local":
        point_ids_t = torch.arange(tracks_t.shape[1], dtype=torch.long, device=device)
    else:
        point_ids_t = torch.as_tensor(keep_idx, dtype=torch.long, device=device)

    if normalize:
        tracks_t[..., 0] = tracks_t[..., 0] / float(max(normalize_width, 1))
        tracks_t[..., 1] = tracks_t[..., 1] / float(max(normalize_height, 1))
        print(
            "[info] normalize_track enabled: dividing track coordinates by "
            f"width={max(normalize_width, 1)} height={max(normalize_height, 1)} "
            "(matched to train_track.py collate logic)."
        )

    return {
        "tracks": tracks_t.unsqueeze(0),
        "visibility": visibility_t.unsqueeze(0),
        "point_mask": point_mask_t.unsqueeze(0),
        "point_ids": point_ids_t.unsqueeze(0),
        "is_normalized": torch.full(
            (1,),
            bool(normalize),
            dtype=torch.bool,
            device=device,
        ),
        "track_resolution": torch.tensor(
            [[float(max(normalize_width, 1)), float(max(normalize_height, 1))]],
            dtype=torch.float32,
            device=device,
        ),
    }


def _load_track_arrays_raw(
    track_file_path: str,
    track_max_points: Optional[int],
    track_point_sample_mode: str,
    track_sort_selected_indices: bool,
    track_point_sample_seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(track_file_path)
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        raise KeyError(f"Track file missing tracks key: {track_file_path}")

    if "visibility_compressed" in data:
        visibility = data["visibility_compressed"]
    elif "visibility" in data:
        visibility = data["visibility"]
    else:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)

    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)

    if track_max_points is not None and int(track_max_points) > 0:
        keep_idx = _select_track_indices(
            num_points=int(tracks.shape[1]),
            max_points=int(track_max_points),
            sample_mode=track_point_sample_mode,
            sort_selected_indices=track_sort_selected_indices,
            seed=track_point_sample_seed,
        )
        tracks = tracks[:, keep_idx]
        visibility = visibility[:, keep_idx]
    return tracks, visibility


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


def _load_clip_feature_npz(
    clip_feature_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    data = np.load(clip_feature_path, allow_pickle=True)
    if "clip_feature" in data:
        clip_feature = data["clip_feature"]
    elif "clip_fea" in data:
        clip_feature = data["clip_fea"]
    else:
        raise KeyError(f"`clip_feature` key not found in: {clip_feature_path}")

    clip_feature_t = torch.as_tensor(clip_feature, dtype=dtype, device=device)
    if clip_feature_t.ndim == 2:
        clip_feature_t = clip_feature_t.unsqueeze(0)
    if clip_feature_t.ndim != 3:
        raise ValueError(
            f"clip_feature must be [L,D] or [B,L,D], got {tuple(clip_feature_t.shape)}"
        )
    return clip_feature_t


def _overlay_tracks_on_video(
    sample: torch.Tensor,
    tracks: np.ndarray,
    visibility: np.ndarray,
    normalize_track: bool,
    normalize_height: int,
    normalize_width: int,
    overlay_linewidth: int = 2,
    overlay_trace_frames: int = -1,
    cotracker_root: Optional[str] = "/data/project-vilab/jaeseok/co-tracker",
    overlay_pad_value: int = 0,
) -> torch.Tensor:
    """Draw tracks on decoded video with co-tracker Visualizer."""
    if sample.ndim != 5:
        return sample
    bsz, ch, num_frames, height, width = sample.shape
    if bsz < 1 or ch != 3:
        return sample

    tracks = np.asarray(tracks, dtype=np.float32).copy()
    visibility = np.asarray(visibility, dtype=np.float32)
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)

    looks_norm = _tracks_look_normalized(tracks)
    # Keep overlay coordinates consistent with model-side mapping:
    # - normalized tracks: x,y in [0,1] -> output pixel coords.
    # - raw tracks: keep pixel coordinates when they already match output size.
    #   Only apply fallback scaling when tracks are clearly out of bounds.
    if looks_norm:
        tracks[..., 0] *= float(max(width - 1, 1))
        tracks[..., 1] *= float(max(height - 1, 1))
    else:
        if normalize_track:
            # These are raw pixel tracks from the same source resolution used for
            # model-side normalization. Preserve out-of-frame coordinates; do not
            # min/max-fit them, or UI-authored trajectories drift on overlay.
            src_w = float(max(normalize_width, 1))
            src_h = float(max(normalize_height, 1))
            tracks[..., 0] = tracks[..., 0] / src_w * float(max(width, 1))
            tracks[..., 1] = tracks[..., 1] / src_h * float(max(height, 1))
            print(
                "[info] overlay: mapped raw pixel tracks by explicit source resolution "
                f"width={src_w:.0f} height={src_h:.0f} -> output width={width} height={height}."
            )
        else:
            x_max = float(np.nanmax(tracks[..., 0])) if tracks.size else 1.0
            x_min = float(np.nanmin(tracks[..., 0])) if tracks.size else 0.0
            y_max = float(np.nanmax(tracks[..., 1])) if tracks.size else 1.0
            y_min = float(np.nanmin(tracks[..., 1])) if tracks.size else 0.0

            within_x = (x_min >= -1.0) and (x_max <= float(width) + 1.0)
            within_y = (y_min >= -1.0) and (y_max <= float(height) + 1.0)
            in_bound_ratio_all = _track_inbound_ratio(
                tracks=tracks,
                visibility=visibility,
                width=width,
                height=height,
            )
            in_bound_ratio_frame0 = _track_inbound_ratio(
                tracks=tracks,
                visibility=visibility,
                width=width,
                height=height,
                frame_slice=slice(0, 1),
            )
            looks_like_stage2_pixel_tracks = (
                (in_bound_ratio_frame0 >= 0.98) or (in_bound_ratio_all >= 0.85)
            )

            if within_x and within_y:
                # Already in output pixel space (e.g., transformed_tracks_grid*_survived.npz).
                pass
            elif looks_like_stage2_pixel_tracks:
                # Out-of-frame points are expected in later frames; keep stage2 pixel coords as-is.
                print(
                    "[info] overlay: keeping raw pixel tracks with partial out-of-frame points "
                    f"(frame0_in_bound={in_bound_ratio_frame0:.4f}, "
                    f"all_in_bound={in_bound_ratio_all:.4f}, normalize_track={normalize_track})."
                )
            else:
                # Fallback: map raw tracks to frame extent.
                x_max = max(x_max, 1.0)
                y_max = max(y_max, 1.0)
                tracks[..., 0] = tracks[..., 0] / x_max * float(max(width - 1, 1))
                tracks[..., 1] = tracks[..., 1] / y_max * float(max(height - 1, 1))
                print(
                    "[info] overlay: fallback-scaled out-of-range raw tracks "
                    f"(x_min={x_min:.2f}, x_max={x_max:.2f}, "
                    f"y_min={y_min:.2f}, y_max={y_max:.2f})."
                )

    frames = (
        sample[0]
        .detach()
        .cpu()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 3, 0)
        .numpy()
    )
    frames_uint8 = (frames * 255.0).round().astype(np.uint8)

    t_steps, num_points, _ = tracks.shape
    draw_frames = min(num_frames, t_steps, visibility.shape[0])
    if draw_frames <= 0:
        return sample

    if cotracker_root is not None and cotracker_root != "":
        cotracker_root_abs = os.path.abspath(cotracker_root)
        if cotracker_root_abs not in sys.path:
            sys.path.insert(0, cotracker_root_abs)
    from cotracker.utils.visualizer import Visualizer

    video_tensor = (
        torch.from_numpy(frames_uint8[:draw_frames]).permute(0, 3, 1, 2).unsqueeze(0).float()
    )
    tracks_tensor = torch.from_numpy(tracks[:draw_frames]).unsqueeze(0).float()
    vis_tensor = torch.from_numpy(visibility[:draw_frames]).unsqueeze(0).float()
    vis = Visualizer(
        save_dir=".",
        pad_value=int(overlay_pad_value),
        linewidth=int(overlay_linewidth),
        fps=16,
        show_first_frame=0,
        tracks_leave_trace=int(overlay_trace_frames),
    )
    vis_out = vis.visualize(
        video=video_tensor,
        tracks=tracks_tensor,
        visibility=vis_tensor,
        filename="noop",
        query_frame=0,
        save_video=False,
    )
    # co-tracker Visualizer returns [B, T, C, H, W]; pipeline expects [B, C, T, H, W].
    out = vis_out.permute(0, 2, 1, 3, 4).contiguous().float() / 255.0
    if bsz > 1:
        keep = sample.detach().cpu().clone()
        keep[0] = out[0]
        return keep
    return out


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


def _resolve_data_path_with_root(
    path_value: str,
    record: Dict,
    data_root: Optional[str],
    root_map: Optional[Dict[str, str]],
    root_id_key: str,
) -> str:
    if os.path.isabs(path_value):
        return path_value
    root_map = root_map or {}
    root_id = record.get(root_id_key, None)
    if root_id is not None:
        root_key = str(root_id).strip()
        if root_key in root_map:
            return os.path.join(root_map[root_key], path_value)
    if data_root is not None:
        return os.path.join(data_root, path_value)
    return os.path.abspath(path_value)


def _extract_first_frame(video_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"first_frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png",
    )

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import PIL.Image

            PIL.Image.fromarray(frame_rgb).save(out_path)
            return out_path
    except Exception:
        pass

    try:
        from torchvision.io import read_video
        import PIL.Image

        frames, _, _ = read_video(video_path, start_pts=0.0, end_pts=1.0, pts_unit="sec")
        if frames is not None and frames.shape[0] > 0:
            first = frames[0].cpu().numpy()
            PIL.Image.fromarray(first).save(out_path)
            return out_path
    except Exception:
        pass

    raise RuntimeError(
        "Failed to extract first frame from video. Install opencv-python or "
        "ensure torchvision video backend is available."
    )


def _select_metadata_index(
    records: List[Dict],
    sample_index: int,
    random_sample: bool,
) -> int:
    if random_sample:
        return random.randint(0, len(records) - 1)

    selected_index = int(sample_index)
    if selected_index < 0:
        selected_index += len(records)
    if selected_index < 0 or selected_index >= len(records):
        raise IndexError(
            f"sample_index={sample_index} out of range for metadata length={len(records)}"
        )
    return selected_index


def _load_root_map(root_map_json: Optional[str]) -> Optional[Dict[str, str]]:
    if root_map_json is None or root_map_json == "":
        return None
    with open(root_map_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"root_map_json must contain object: {root_map_json}")
    return {str(k): str(v) for k, v in loaded.items()}


def _resolve_track_from_metadata_index(
    metadata_path: str,
    metadata_index: int,
    train_data_dir: Optional[str],
    root_map_json: Optional[str],
    root_id_key: str,
) -> Tuple[Dict, str]:
    records = _read_metadata_records(metadata_path)
    if len(records) == 0:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    target_index = int(metadata_index)
    if target_index < 0:
        target_index += len(records)
    if target_index < 0 or target_index >= len(records):
        raise IndexError(
            f"track metadata index {metadata_index} out of range for metadata length={len(records)}"
        )

    record = records[target_index]
    root_map = _load_root_map(root_map_json)

    track_file_path = record.get("track_file_path", None)
    if not track_file_path:
        raise KeyError(
            f"Metadata row index={target_index} has no `track_file_path`."
        )
    resolved_track = _resolve_data_path_with_root(
        str(track_file_path), record, train_data_dir, root_map, root_id_key
    )
    if not os.path.isfile(resolved_track):
        raise FileNotFoundError(f"Track file not found from metadata: {resolved_track}")
    return record, resolved_track


def _resolve_sample_from_metadata(
    metadata_path: str,
    sample_index: int,
    random_sample: bool,
    train_data_dir: Optional[str],
    root_map_json: Optional[str],
    root_id_key: str,
) -> Tuple[Dict, int, str, Optional[str]]:
    records = _read_metadata_records(metadata_path)
    if len(records) == 0:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    selected_index = _select_metadata_index(
        records=records,
        sample_index=sample_index,
        random_sample=random_sample,
    )
    record = records[selected_index]

    root_map = _load_root_map(root_map_json)

    track_file_path = record.get("track_file_path", None)
    if not track_file_path:
        raise KeyError("Selected metadata row has no `track_file_path`.")
    resolved_track = _resolve_data_path_with_root(
        str(track_file_path), record, train_data_dir, root_map, root_id_key
    )
    if not os.path.isfile(resolved_track):
        raise FileNotFoundError(f"Track file not found from metadata: {resolved_track}")

    # Prefer explicit first-frame field if present.
    first_frame_keys = ["first_frame_path", "image_path", "frame_path"]
    for key in first_frame_keys:
        candidate = record.get(key, None)
        if candidate:
            resolved_image = _resolve_data_path_with_root(
                str(candidate), record, train_data_dir, root_map, root_id_key
            )
            if os.path.isfile(resolved_image):
                return record, selected_index, resolved_track, resolved_image

    media_path_value = record.get("file_path", None)
    if media_path_value is None:
        raise KeyError("Selected metadata row has no `file_path`.")
    media_path = _resolve_data_path_with_root(
        str(media_path_value), record, train_data_dir, root_map, root_id_key
    )

    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if media_path.lower().endswith(image_exts) and os.path.isfile(media_path):
        return record, selected_index, resolved_track, media_path

    # Common latent-preprocess layout: first frame image lives beside vae_latents.pt.
    media_dir = os.path.dirname(media_path)
    first_frame_candidates = [
        os.path.join(media_dir, "first_frame.png"),
        os.path.join(media_dir, "first_frame.jpg"),
        os.path.join(media_dir, "first_frame.jpeg"),
        os.path.join(media_dir, "first_frame.webp"),
        os.path.join(media_dir, "first_frame.bmp"),
    ]
    for first_frame_path in first_frame_candidates:
        if os.path.isfile(first_frame_path):
            return record, selected_index, resolved_track, first_frame_path

    # Latent metadata usually points to vae_latents.pt; infer nearby decoded/final mp4.
    search_video_candidates = [media_path]
    search_video_candidates.extend(
        [
            os.path.join(media_dir, "decoded_from_vae_832x480_16fps_81f.mp4"),
            os.path.join(media_dir, "final_832x480_16fps_81f.mp4"),
        ]
    )

    for video_path in search_video_candidates:
        if os.path.isfile(video_path) and video_path.lower().endswith(".mp4"):
            temp_dir = tempfile.mkdtemp(prefix="wan_track_first_frame_")
            first_frame_path = _extract_first_frame(video_path, temp_dir)
            return record, selected_index, resolved_track, first_frame_path

    raise FileNotFoundError(
        "Could not resolve first frame image from metadata row. "
        f"Checked media path: {media_path}"
    )


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _want_pdb_track(args: argparse.Namespace) -> bool:
    """Interactive pdb before pipeline (needs a TTY; disabled if python -O)."""
    if getattr(args, "pdb_track_condition", False):
        return True
    return _env_truthy("PDB_TRACK_CONDITION")


def _want_pdb_pipeline_step0(args: argparse.Namespace) -> bool:
    """First denoise step inside WanFunInpaintPipeline (env or flag)."""
    if getattr(args, "pdb_pipeline_step0", False):
        return True
    return _env_truthy("PDB_PIPELINE_STEP0")


def _debug_log_track_dict(prefix: str, tc: Optional[Dict[str, torch.Tensor]]) -> None:
    """Print track_condition tensors for manual tracing (use with --debug_track_condition)."""
    if tc is None:
        print(f"[debug_track] {prefix}: track_condition is None")
        return
    print(f"[debug_track] {prefix}: keys={sorted(tc.keys())}")
    for k in sorted(tc.keys()):
        v = tc[k]
        if not isinstance(v, torch.Tensor):
            print(f"  {k}: {type(v)} (non-tensor)")
            continue
        vf = v.detach().float()
        extra = ""
        if k == "visibility":
            extra = f" frac>0.5={(vf > 0.5).float().mean().item():.4f}"
        elif k == "tracks":
            extra = (
                f" x[{vf[..., 0].min().item():.4f},{vf[..., 0].max().item():.4f}]"
                f" y[{vf[..., 1].min().item():.4f},{vf[..., 1].max().item():.4f}]"
            )
        elif k == "point_mask":
            extra = f" true_count={int(v.sum().item())}/{v.numel()}"
        elif k == "is_normalized":
            extra = f" values={v.detach().cpu().tolist()}"
        elif k == "track_resolution":
            extra = f" values={v.detach().cpu().tolist()}"
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}{extra}")


def _trace_print(prefix: str, message: str) -> None:
    print(f"[trace_track] {prefix}: {message}")

def _append_track_analysis_summary(
    save_dir: str,
    timestamp: str,
    args: argparse.Namespace,
    selected_metadata: Optional[Dict],
    resolved_track_file_path: Optional[str],
    transformer: WanTransformer3DModelTrack,
) -> None:
    if not getattr(args, "track_analysis", False):
        return
    if not hasattr(transformer, "get_last_track_analysis"):
        return

    metrics = transformer.get_last_track_analysis()
    if not metrics:
        return

    row: Dict[str, object] = {
        "timestamp": timestamp,
        "sample_index": int(getattr(args, "sample_index", 0)),
        "track_file_path": str(resolved_track_file_path or ""),
        "prompt_head": str(args.prompt)[:120],
    }
    if selected_metadata is not None:
        row["metadata_file_path"] = str(selected_metadata.get("file_path", ""))
        row["metadata_track_file_path"] = str(selected_metadata.get("track_file_path", ""))
    for key, val in metrics.items():
        row[key] = float(val)

    json_path = os.path.join(save_dir, f"track_analysis_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(save_dir, "track_analysis_summary.csv")
    fieldnames = list(row.keys())
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[done] saved track analysis json: {json_path}")
    print(f"[done] appended track analysis csv: {csv_path}")


def _make_random_fake_track_condition(
    track_condition: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    fake = dict(track_condition)
    tracks = track_condition["tracks"]
    track_resolution = track_condition.get("track_resolution", None)
    visibility = track_condition.get("visibility", None)

    if track_resolution is not None:
        track_resolution = track_resolution.to(device=tracks.device, dtype=torch.float32)
        if track_resolution.ndim == 1:
            track_resolution = track_resolution.unsqueeze(0)
        while track_resolution.shape[0] < tracks.shape[0]:
            track_resolution = track_resolution.expand(tracks.shape[0], -1)
        src_w = torch.clamp(track_resolution[:, 0].view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(track_resolution[:, 1].view(-1, 1, 1), min=1.0)
    else:
        src_w = torch.full(
            (tracks.shape[0], 1, 1),
            float(max(tracks.shape[-2], 1)),
            device=tracks.device,
            dtype=torch.float32,
        )
        src_h = torch.full(
            (tracks.shape[0], 1, 1),
            float(max(tracks.shape[-3], 1)),
            device=tracks.device,
            dtype=torch.float32,
        )

    fake_tracks = torch.empty_like(tracks)
    fake_tracks[..., 0] = torch.rand_like(tracks[..., 0]) * src_w
    fake_tracks[..., 1] = torch.rand_like(tracks[..., 1]) * src_h
    fake["tracks"] = fake_tracks
    if visibility is not None:
        fake["visibility"] = visibility.clone()
    point_mask = track_condition.get("point_mask", None)
    if point_mask is not None:
        fake["point_mask"] = point_mask.clone()
    if track_resolution is not None:
        fake["track_resolution"] = track_resolution.clone()
    return fake


def _as_batched_bool_hint(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    hint = (
        value.to(device=device)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=device)
    )
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
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    resolution = (
        value.to(device=device, dtype=torch.float32)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=device, dtype=torch.float32)
    )
    if resolution.ndim == 1:
        resolution = resolution.unsqueeze(0)
    if resolution.ndim != 2 or resolution.shape[-1] < 2:
        raise ValueError(
            f"track_resolution must be [B,2] or [2], got {tuple(resolution.shape)}"
        )
    resolution = resolution[:, :2]
    if resolution.shape[0] == 1 and batch_size > 1:
        resolution = resolution.expand(batch_size, -1)
    if resolution.shape[0] != batch_size:
        raise ValueError(
            f"track_resolution batch {resolution.shape[0]} != tracks batch {batch_size}"
        )
    return resolution


def _map_wan_move_tracks_to_latent_grid(
    tracks: torch.Tensor,
    h_lat: int,
    w_lat: int,
    is_normalized_hint: Optional[torch.Tensor],
    track_resolution: Optional[torch.Tensor],
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
        src_w = torch.clamp(x.amax(dim=(1, 2)).view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(y.amax(dim=(1, 2)).view(-1, 1, 1), min=1.0)
    else:
        src_w = torch.clamp(resolution[:, 0].view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(resolution[:, 1].view(-1, 1, 1), min=1.0)

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
) -> torch.Tensor:
    if track_condition is None:
        return condition_latents

    tracks = track_condition.get("tracks", None)
    visibility = track_condition.get("visibility", None)
    if tracks is None or visibility is None:
        return condition_latents

    device = condition_latents.device
    bsz, _, latent_frames, h_lat, w_lat = condition_latents.shape
    tracks = tracks.to(device=device, dtype=torch.float32)
    visibility = visibility.to(device=device, dtype=torch.float32)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be [B,T,P,2], got {tuple(tracks.shape)}")
    if visibility.ndim != 3:
        raise ValueError(f"visibility must be [B,T,P], got {tuple(visibility.shape)}")
    if tracks.shape[0] != bsz:
        raise ValueError(
            f"Batch mismatch between condition latents ({bsz}) and tracks ({tracks.shape[0]})."
        )
    if visibility.shape[:3] != tracks.shape[:3]:
        raise ValueError(
            "tracks/visibility shape mismatch: "
            f"tracks={tuple(tracks.shape)} visibility={tuple(visibility.shape)}"
        )

    if latent_frames <= 1:
        return condition_latents

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

    source_valid = valid[:, 0, :]
    target_valid = valid[:, 1:, :] & source_valid[:, None, :]
    valid_indices = target_valid.nonzero(as_tuple=False)
    if valid_indices.numel() == 0:
        return condition_latents

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
    return edited


def _attach_wan_move_forward_adapter(
    transformer: WanTransformer3DModel,
    latent_channels: int,
    temporal_stride: int,
) -> None:
    base_forward = transformer.forward
    latent_channels = int(latent_channels)
    temporal_stride = int(max(1, temporal_stride))

    def _forward_with_wan_move(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        track_condition: Optional[Dict[str, torch.Tensor]] = None,
    ):
        y_input = y
        if y_input is not None and track_condition is not None:
            if y_input.ndim != 5:
                raise ValueError(f"Expected y to be 5D [B,C,F,H,W], got {tuple(y_input.shape)}")
            if y_input.shape[1] < latent_channels:
                raise ValueError(
                    f"y channels {y_input.shape[1]} smaller than latent_channels={latent_channels}"
                )
            y_track = y_input[:, -latent_channels:, :, :, :]
            y_track_wan_move = _apply_wan_move_feature_replace(
                condition_latents=y_track,
                track_condition=track_condition,
                temporal_stride=temporal_stride,
            )
            if y_track_wan_move is not y_track:
                y_input = y_input.clone()
                y_input[:, -latent_channels:, :, :, :] = y_track_wan_move

        return base_forward(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y_input,
            y_camera=y_camera,
            full_ref=full_ref,
            subject_ref=subject_ref,
            cond_flag=cond_flag,
        )

    transformer.forward = types.MethodType(_forward_with_wan_move, transformer)
    setattr(transformer, "_wan_move_adapter_enabled", True)
    setattr(transformer, "_wan_move_temporal_stride", temporal_stride)
    setattr(transformer, "_wan_move_latent_channels", latent_channels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.1 Fun Track i2v inference")
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
    parser.add_argument("--validation_image_start", type=str, default=None)
    parser.add_argument("--validation_image_end", type=str, default=None)
    parser.add_argument(
        "--track_file_path",
        type=str,
        default=None,
        help="Optional .npz track file path used during track-conditioned inference.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Optional training metadata path (json/jsonl/csv). If set, one sample is used.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Metadata sample index to use when --metadata_path is set.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Use a random row from metadata (overrides --sample_index).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Equivalent to training --train_data_dir; used to resolve relative metadata paths.",
    )
    parser.add_argument(
        "--train_data_root_map_json_track",
        type=str,
        default=None,
        help="Optional root_id->abs_root json map for multi-root metadata.",
    )
    parser.add_argument(
        "--train_data_root_id_key_track",
        type=str,
        default="root_id",
        help="Metadata key name for root id lookup.",
    )
    parser.add_argument(
        "--use_prompt_from_metadata",
        action="store_true",
        help="Use selected metadata row `text` as prompt.",
    )
    parser.add_argument(
        "--track_condition_index_offset",
        type=int,
        default=0,
        help=(
            "When --metadata_path is used, keep text/first-frame at base sample index, "
            "but load track condition from (sample_index + offset). "
            "Example: offset=1 uses next row track for mismatch testing."
        ),
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
    parser.add_argument(
        "--clip_feature_path",
        type=str,
        default=None,
        help="Optional precomputed CLIP feature npz with `clip_feature`.",
    )
    parser.add_argument(
        "--zero_clip_context",
        action="store_true",
        help=(
            "Force zero CLIP context during sampling by ignoring both external "
            "clip_feature input and first-frame clip_image conditioning."
        ),
    )
    parser.add_argument("--sample_height", type=int, default=480)
    parser.add_argument("--sample_width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="cfg",
        choices=["cfg", "joint_tm", "text_only", "motion_only", "unified"],
        help=(
            "Guidance mode. `cfg` uses standard classifier-free guidance; "
            "`joint_tm` uses text-motion joint guidance; "
            "`text_only` nulls motion on all branches; "
            "`motion_only` nulls text on all branches; "
            "`unified` uses Wan-Move style CFG with text+motion vs null_text+null_motion."
        ),
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--text_guidance_weight",
        type=float,
        default=3.0,
        help="Text guidance weight used by joint_tm/text_only modes.",
    )
    parser.add_argument(
        "--motion_guidance_weight",
        type=float,
        default=1.5,
        help="Motion guidance weight used by joint_tm/motion_only modes.",
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
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--track_head_hidden_dim",
        type=int,
        default=None,
        help=(
            "Override track head hidden dim for loading checkpoints trained with a "
            "non-default hidden width."
        ),
    )
    parser.add_argument("--save_dir", type=str, default="samples/wan-videos-fun-i2v-track")
    parser.add_argument(
        "--output_name_suffix",
        type=str,
        default="",
        help="Optional suffix appended to saved output filenames for compare experiments.",
    )
    parser.add_argument("--normalize_track", action="store_true")
    parser.add_argument("--track_normalize_height", type=int, default=480)
    parser.add_argument("--track_normalize_width", type=int, default=832)
    parser.add_argument(
        "--track_max_points",
        type=int,
        default=-1,
        help="Max track points to use (>0 to limit, <=0 to use all points).",
    )
    parser.add_argument(
        "--track_point_sample_mode",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
        help=(
            "Track point sampling mode when --track_max_points > 0. "
            "uniform=deterministic evenly spaced indices (default), "
            "random=random subset."
        ),
    )
    parser.add_argument(
        "--track_sort_selected_indices",
        type=_str2bool,
        default=True,
        help=(
            "Only for --track_point_sample_mode=random. "
            "If true, sampled indices are sorted to preserve original order."
        ),
    )
    parser.add_argument(
        "--track_point_sample_seed",
        type=int,
        default=None,
        help=(
            "Optional seed for random track point sampling. "
            "If unset, --seed is used."
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
    parser.add_argument(
        "--overlay_linewidth",
        type=int,
        default=2,
        help="Track line width for overlay video (co-tracker style).",
    )
    parser.add_argument(
        "--overlay_trace_frames",
        type=int,
        default=-1,
        help="Frames of trajectory history per point (-1 = full history, like cotracker trace).",
    )
    parser.add_argument(
        "--cotracker_root",
        type=str,
        default="/data/project-vilab/jaeseok/co-tracker",
        help="Path to co-tracker repo root for importing cotracker.utils.visualizer.",
    )
    parser.add_argument(
        "--overlay_pad_value",
        type=int,
        default=0,
        help="Pad value passed to co-tracker Visualizer.",
    )
    parser.add_argument(
        "--debug_track_condition",
        action="store_true",
        help="Print only: track_condition after load + pipeline step-0 shapes (does not start pdb).",
    )
    parser.add_argument(
        "--track_analysis",
        action="store_true",
        help="Enable step-0 quantitative analysis for track signal strength and early-block retention.",
    )
    parser.add_argument(
        "--pdb_track_condition",
        action="store_true",
        help="Stop in pdb once before pipeline() (or set env PDB_TRACK_CONDITION=true). Requires an interactive terminal.",
    )
    parser.add_argument(
        "--pdb_pipeline_step0",
        action="store_true",
        help="Stop in pdb at pipeline denoise step 0, right before transformer (env PDB_PIPELINE_STEP0=true).",
    )
    parser.add_argument(
        "--force_track_condition_none",
        action="store_true",
        help="Force track_condition=None before pipeline() for A/B comparison with identical inputs/seed.",
    )
    parser.add_argument(
        "--random_fake_track",
        action="store_true",
        help="Replace loaded track coordinates with random fake trajectories while preserving shape/visibility.",
    )
    parser.add_argument(
        "--track_latent_scale",
        type=float,
        default=1.0,
        help="Inference-only scale factor applied to track latent before concat.",
    )
    parser.add_argument(
        "--track_condition_mode",
        type=str,
        default="track_head",
        choices=["track_head", "wan_move"],
        help=(
            "track_head uses the track-head transformer checkpoint; "
            "wan_move uses base Wan transformer and applies Wan-Move latent copy by tracks."
        ),
    )
    parser.add_argument(
        "--wan_move_temporal_stride",
        type=int,
        default=0,
        help=(
            "Temporal stride for mapping video-track frames to latent frames in wan_move mode. "
            "<=0 uses VAE temporal_compression_ratio."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(float(args.track_latent_scale)):
        raise ValueError("--track_latent_scale must be finite.")
    if float(args.track_latent_scale) < 0.0:
        raise ValueError("--track_latent_scale must be >= 0.")
    effective_track_sample_seed = (
        args.seed if args.track_point_sample_seed is None else args.track_point_sample_seed
    )
    os.makedirs(args.save_dir, exist_ok=True)
    _trace_print(
        "args",
        (
            f"validation_image_start={args.validation_image_start} "
            f"track_file_path={args.track_file_path} "
            f"metadata_path={args.metadata_path} "
            f"track_condition_index_offset={args.track_condition_index_offset} "
            f"track_point_sample_mode={args.track_point_sample_mode} "
            f"track_sort_selected_indices={args.track_sort_selected_indices} "
            f"track_point_sample_seed={effective_track_sample_seed} "
            f"track_point_id_mode={args.track_point_id_mode} "
            f"guidance_mode={args.guidance_mode} "
            f"guidance_scale={args.guidance_scale} "
            f"text_guidance_weight={args.text_guidance_weight} "
            f"motion_guidance_weight={args.motion_guidance_weight} "
            f"debug_track_condition={args.debug_track_condition} "
            f"track_analysis={args.track_analysis} "
            f"force_track_condition_none={args.force_track_condition_none} "
            f"random_fake_track={args.random_fake_track} "
            f"track_latent_scale={args.track_latent_scale} "
            f"track_head_hidden_dim={args.track_head_hidden_dim} "
            f"track_condition_mode={args.track_condition_mode} "
            f"wan_move_temporal_stride={args.wan_move_temporal_stride}"
        ),
    )

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

    transformer_additional_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    if args.track_condition_mode == "track_head":
        if args.track_head_hidden_dim is not None:
            if int(args.track_head_hidden_dim) <= 0:
                raise ValueError("--track_head_hidden_dim must be > 0 when provided.")
            transformer_additional_kwargs["track_head_hidden_dim"] = int(args.track_head_hidden_dim)
        transformer_additional_kwargs["track_latent_scale"] = float(args.track_latent_scale)
        transformer_cls = WanTransformer3DModelTrack
    else:
        transformer_cls = WanTransformer3DModel
        if args.track_head_hidden_dim is not None:
            print(
                "[warn] --track_head_hidden_dim is ignored when "
                "--track_condition_mode=wan_move."
            )
        if float(args.track_latent_scale) != 1.0:
            print(
                "[warn] --track_latent_scale is only used by track_head mode; "
                "wan_move mode ignores it."
            )

    transformer = transformer_cls.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
        ),
        transformer_additional_kwargs=transformer_additional_kwargs,
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
    if args.track_condition_mode == "wan_move":
        wan_move_temporal_stride = int(args.wan_move_temporal_stride)
        if wan_move_temporal_stride <= 0:
            wan_move_temporal_stride = int(getattr(vae.config, "temporal_compression_ratio", 4))
        _attach_wan_move_forward_adapter(
            transformer=transformer,
            latent_channels=int(getattr(vae.config, "latent_channels", 16)),
            temporal_stride=wan_move_temporal_stride,
        )
        print(
            "[info] wan_move adapter enabled: "
            f"latent_channels={int(getattr(vae.config, 'latent_channels', 16))} "
            f"temporal_stride={wan_move_temporal_stride}"
        )

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

    generator = torch.Generator(device=device).manual_seed(args.seed)
    video_length = int(
        (args.video_length - 1) // vae.config.temporal_compression_ratio
        * vae.config.temporal_compression_ratio
    ) + 1 if args.video_length != 1 else 1

    selected_metadata = None
    selected_metadata_index = None
    resolved_validation_image_start = args.validation_image_start
    resolved_track_file_path = args.track_file_path
    _trace_print(
        "resolve_inputs:init",
        f"resolved_validation_image_start={resolved_validation_image_start} resolved_track_file_path={resolved_track_file_path}",
    )

    if args.metadata_path is not None and args.metadata_path != "":
        (
            selected_metadata,
            selected_metadata_index,
            resolved_track_file_path,
            resolved_validation_image_start,
        ) = _resolve_sample_from_metadata(
            metadata_path=args.metadata_path,
            sample_index=args.sample_index,
            random_sample=args.random_sample,
            train_data_dir=args.train_data_dir,
            root_map_json=args.train_data_root_map_json_track,
            root_id_key=args.train_data_root_id_key_track,
        )
        print(
            "[info] metadata sample selected: "
            f"track={resolved_track_file_path}, first_frame={resolved_validation_image_start}"
        )
        if args.use_prompt_from_metadata:
            meta_prompt = str(selected_metadata.get("text", "")).strip()
            if meta_prompt != "":
                args.prompt = meta_prompt
                print("[info] prompt loaded from metadata `text`.")
        _trace_print(
            "resolve_inputs:metadata",
            f"resolved_validation_image_start={resolved_validation_image_start} resolved_track_file_path={resolved_track_file_path}",
        )

    if int(args.track_condition_index_offset) != 0:
        if args.metadata_path is None or args.metadata_path == "":
            raise ValueError(
                "--track_condition_index_offset requires --metadata_path "
                "(it shifts metadata row only for track condition)."
            )
        if selected_metadata_index is None:
            raise RuntimeError("internal error: selected_metadata_index is not set.")

        target_track_index = int(selected_metadata_index) + int(args.track_condition_index_offset)
        _, shifted_track_path = _resolve_track_from_metadata_index(
            metadata_path=args.metadata_path,
            metadata_index=target_track_index,
            train_data_dir=args.train_data_dir,
            root_map_json=args.train_data_root_map_json_track,
            root_id_key=args.train_data_root_id_key_track,
        )
        print(
            "[info] applying track_condition index offset: "
            f"base_index={selected_metadata_index}, "
            f"offset={args.track_condition_index_offset}, "
            f"track_index={target_track_index}, "
            f"track_path={shifted_track_path}"
        )
        resolved_track_file_path = shifted_track_path

    if resolved_validation_image_start is None:
        raise ValueError(
            "validation_image_start is required unless --metadata_path can resolve first frame."
        )
    if (args.text_feature_path is None or args.text_feature_path == "") and args.prompt.strip() == "":
        raise ValueError(
            "prompt is empty. Pass --prompt or set --use_prompt_from_metadata with metadata row text."
        )

    input_video, input_video_mask, clip_image = get_image_to_video_latent(
        resolved_validation_image_start,
        args.validation_image_end,
        video_length=video_length,
        sample_size=[args.sample_height, args.sample_width],
    )
    if args.zero_clip_context:
        if clip_image is not None:
            print("[info] zero_clip_context enabled: dropping first-frame clip_image.")
        clip_image = None
    _trace_print(
        "latent_inputs",
        (
            f"input_video_shape={tuple(input_video.shape)} "
            f"input_video_mask_shape={tuple(input_video_mask.shape)} "
            f"clip_image_present={clip_image is not None}"
        ),
    )

    track_condition = None
    if resolved_track_file_path is not None and resolved_track_file_path != "":
        _trace_print(
            "load_track_condition:before",
            (
                f"path={resolved_track_file_path} exists={os.path.isfile(resolved_track_file_path)} "
                f"normalize={args.normalize_track} "
                f"max_points={args.track_max_points} "
                f"sample_mode={args.track_point_sample_mode} "
                f"sort_selected_indices={args.track_sort_selected_indices} "
                f"sample_seed={effective_track_sample_seed} "
                f"point_id_mode={args.track_point_id_mode}"
            ),
        )
        track_condition = _load_track_condition(
            track_file_path=resolved_track_file_path,
            normalize=args.normalize_track,
            normalize_height=args.track_normalize_height,
            normalize_width=args.track_normalize_width,
            track_max_points=args.track_max_points,
            track_point_sample_mode=args.track_point_sample_mode,
            track_sort_selected_indices=args.track_sort_selected_indices,
            track_point_sample_seed=effective_track_sample_seed,
            track_point_id_mode=args.track_point_id_mode,
            device=device,
        )
        if os.environ.get("PDB_DEBUG", "0") == "1":
            import pdb; pdb.set_trace()
        _trace_print(
            "load_track_condition:after",
            f"type={type(track_condition).__name__} is_none={track_condition is None}",
        )
        print(
            "[info] track_condition loaded: "
            f"tracks={tuple(track_condition['tracks'].shape)}, "
            f"visibility={tuple(track_condition['visibility'].shape)}"
        )
    else:
        _trace_print("load_track_condition:skip", "resolved_track_file_path is empty")

    if args.debug_track_condition:
        _debug_log_track_dict("predict_i2v_track: after load (before pipeline)", track_condition)

    if args.random_fake_track and track_condition is not None:
        _trace_print(
            "track_compare",
            "random_fake_track=true -> replacing loaded track coordinates with random fake trajectories",
        )
        track_condition = _make_random_fake_track_condition(track_condition)
        if args.debug_track_condition:
            _debug_log_track_dict("predict_i2v_track: fake track (before pipeline)", track_condition)

    if args.force_track_condition_none:
        _trace_print(
            "track_compare",
            "force_track_condition_none=true -> overriding loaded track_condition to None before pipeline()",
        )
        track_condition = None

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
        # Fallback: use zero tensor as unconditional embedding.
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError(
                "negative prompt embeds shape mismatch: "
                f"{tuple(negative_prompt_embeds.shape)} vs {tuple(prompt_embeds.shape)}"
            )

    clip_feature = None
    if args.zero_clip_context:
        if args.clip_feature_path is not None and args.clip_feature_path != "":
            print(
                "[info] zero_clip_context enabled: ignoring external clip feature "
                f"path={args.clip_feature_path}"
            )
    elif args.clip_feature_path is not None and args.clip_feature_path != "":
        clip_feature = _load_clip_feature_npz(
            clip_feature_path=args.clip_feature_path,
            device=device,
            dtype=weight_dtype,
        )
        print(f"[info] loaded external clip feature: {tuple(clip_feature.shape)}")

    if args.debug_track_condition:
        os.environ["WAN_DEBUG_TRACK_CONDITION"] = "1"
    if args.track_analysis:
        os.environ["WAN_TRACK_ANALYSIS"] = "1"
    os.environ["WAN_TRACK_LATENT_SCALE"] = str(args.track_latent_scale)

    if _want_pdb_pipeline_step0(args):
        os.environ["PDB_PIPELINE_STEP0"] = "1"

    if _want_pdb_track(args):
        print(
            "[pdb] Stopping before pipeline(): inspect track_condition, prompt_embeds, "
            "input_video, …  (continue with `c`, step with `n`, quit with `q`)",
            flush=True,
        )
        breakpoint()

    try:
        with torch.no_grad():
            _trace_print(
                "pipeline_call",
                (
                    f"track_condition_is_none={track_condition is None} "
                    f"prompt_embeds_is_none={prompt_embeds is None} "
                    f"negative_prompt_embeds_is_none={negative_prompt_embeds is None} "
                    f"clip_feature_is_none={clip_feature is None} "
                    f"clip_image_present={clip_image is not None} "
                    f"zero_clip_context={args.zero_clip_context}"
                ),
            )
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
        if args.debug_track_condition:
            os.environ.pop("WAN_DEBUG_TRACK_CONDITION", None)
        if args.track_analysis:
            os.environ.pop("WAN_TRACK_ANALYSIS", None)
        os.environ.pop("WAN_TRACK_LATENT_SCALE", None)
        if getattr(args, "pdb_pipeline_step0", False):
            os.environ.pop("PDB_PIPELINE_STEP0", None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _append_track_analysis_summary(
        save_dir=args.save_dir,
        timestamp=timestamp,
        args=args,
        selected_metadata=selected_metadata,
        resolved_track_file_path=resolved_track_file_path,
        transformer=transformer,
    )
    suffix = str(args.output_name_suffix).strip()
    suffix_part = f"_{suffix}" if suffix != "" else ""
    output_plain = os.path.join(args.save_dir, f"track_i2v{suffix_part}_{timestamp}.mp4")
    save_videos_grid(sample, output_plain, fps=args.fps)
    print(f"[done] saved plain video: {output_plain}")

    if resolved_track_file_path is not None and resolved_track_file_path != "":
        raw_tracks, raw_visibility = _load_track_arrays_raw(
            resolved_track_file_path,
            track_max_points=args.track_max_points,
            track_point_sample_mode=args.track_point_sample_mode,
            track_sort_selected_indices=args.track_sort_selected_indices,
            track_point_sample_seed=effective_track_sample_seed,
        )
        sample_overlay = _overlay_tracks_on_video(
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
        save_videos_grid(sample_overlay, output_overlay, fps=args.fps)
        print(f"[done] saved track overlay video: {output_overlay}")


if __name__ == "__main__":
    main()
