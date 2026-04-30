#!/usr/bin/env python
"""Visualize track-to-latent canvas mapping before training.

This script reproduces the core mapping/scatter logic used in
`WanTransformer3DModelTrack` and helps diagnose whether track points collapse
to a tiny region because of coordinate-scale mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


@dataclass
class ModeResult:
    mode: str
    decision_is_normalized: Optional[bool]
    canvas: torch.Tensor  # [B, T, H, W]
    nonzero_sites: int
    total_sites: int
    nonzero_ratio: float
    active_cells_union: int
    visible_points: int
    visible_inbound_points: int
    total_hits: int
    collision_cells: int
    collision_cell_ratio: float
    collision_hits: int
    collision_hit_ratio: float
    max_cell_count: int


def _apply_latent_axis_style(
    ax: plt.Axes,
    latent_h: int,
    latent_w: int,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, float(max(latent_w, 1)) - 0.5)
    ax.set_ylim(float(max(latent_h, 1)) - 0.5, -0.5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))


def _apply_pixel_axis_style(
    ax: plt.Axes,
    display_h: int,
    display_w: int,
) -> None:
    ax.set_xlim(0.0, float(max(display_w - 1, 1)))
    ax.set_ylim(float(max(display_h - 1, 1)), 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize track canvas mapping for debugging."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--track_npz",
        nargs="+",
        default=None,
        help="Track npz file path(s). Can pass multiple files as one batch.",
    )
    source_group.add_argument(
        "--meta_json",
        type=str,
        default=None,
        help="Metadata json path containing rows with track file paths.",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional root path prepended to relative track paths from --meta_json.",
    )
    parser.add_argument(
        "--track_key",
        type=str,
        default="track_file_path",
        help="Track path key used in --meta_json rows.",
    )
    parser.add_argument(
        "--meta_offset",
        type=int,
        default=0,
        help="Start index for --meta_json sampling.",
    )
    parser.add_argument(
        "--meta_count",
        type=int,
        default=8,
        help="Number of rows to load from --meta_json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir_wan2.1_fun_track/track_canvas_debug",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--latent_h",
        type=int,
        default=60,
        help="Latent height used by track-to-grid mapping.",
    )
    parser.add_argument(
        "--latent_w",
        type=int,
        default=104,
        help="Latent width used by track-to-grid mapping.",
    )
    parser.add_argument(
        "--track_resolution_h",
        type=int,
        default=480,
        help="Source frame height used for track_resolution.",
    )
    parser.add_argument(
        "--track_resolution_w",
        type=int,
        default=832,
        help="Source frame width used for track_resolution.",
    )
    parser.add_argument(
        "--apply_track_normalize",
        action="store_true",
        help="Apply collate-style normalization: x/=W, y/=H before model mapping.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["auto", "normalized"],
        choices=["auto", "normalized", "pixel"],
        help=(
            "Mapping mode(s): auto reproduces model heuristic, "
            "normalized forces normalized branch, pixel forces pixel branch."
        ),
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index (from sample 0) to render detailed heatmap.",
    )
    parser.add_argument(
        "--save_mp4",
        action="store_true",
        help="Save frame-by-frame visualization as MP4.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="FPS for generated MP4.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum frames to render for MP4. -1 means all frames.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Batch sample index used for per-frame track/canvas MP4.",
    )
    parser.add_argument(
        "--heat_percentile",
        type=float,
        default=99.5,
        help=(
            "Upper percentile for heat clipping before normalization. "
            "Smaller values make medium-density areas brighter."
        ),
    )
    parser.add_argument(
        "--heat_gamma",
        type=float,
        default=0.6,
        help=(
            "Gamma after normalization. Values < 1 brighten dark areas."
        ),
    )
    return parser.parse_args()


def load_track_npz(path: str) -> Dict[str, torch.Tensor]:
    data = np.load(path)
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        raise KeyError(f"Missing track key in npz: {path}")

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

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Unexpected tracks shape {tracks.shape} from {path}")
    if visibility.ndim != 2:
        raise ValueError(f"Unexpected visibility shape {visibility.shape} from {path}")

    return {
        "tracks": torch.as_tensor(tracks, dtype=torch.float32),
        "visibility": torch.as_tensor(visibility, dtype=torch.float32),
    }


def resolve_meta_track_paths(
    meta_json: str,
    data_root: Optional[str],
    track_key: str,
    offset: int,
    count: int,
) -> List[str]:
    with open(meta_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in metadata json: {meta_json}")

    selected = rows[offset : offset + count]
    if len(selected) == 0:
        raise ValueError(
            f"No rows selected from metadata: offset={offset}, count={count}, total={len(rows)}"
        )

    out_paths: List[str] = []
    for i, row in enumerate(selected):
        if not isinstance(row, dict):
            raise ValueError(f"Metadata row {offset + i} is not dict.")
        rel_path = row.get(track_key, None)
        if rel_path is None or str(rel_path).strip() == "":
            raise KeyError(f"Missing `{track_key}` at row {offset + i}.")
        rel_path = str(rel_path)
        if os.path.isabs(rel_path):
            full_path = rel_path
        elif data_root is not None:
            full_path = os.path.join(data_root, rel_path)
        else:
            full_path = os.path.join(os.path.dirname(meta_json), rel_path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Track file not found: {full_path}")
        out_paths.append(full_path)
    return out_paths


def pad_track_batch(
    items: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    bsz = len(items)
    t_max = max(int(item["tracks"].shape[0]) for item in items)
    p_max = max(int(item["tracks"].shape[1]) for item in items)

    tracks = torch.zeros((bsz, t_max, p_max, 2), dtype=torch.float32)
    visibility = torch.zeros((bsz, t_max, p_max), dtype=torch.float32)
    point_mask = torch.zeros((bsz, p_max), dtype=torch.bool)

    for i, item in enumerate(items):
        cur_t = int(item["tracks"].shape[0])
        cur_p = int(item["tracks"].shape[1])
        tracks[i, :cur_t, :cur_p] = item["tracks"]
        visibility[i, :cur_t, :cur_p] = item["visibility"]
        point_mask[i, :cur_p] = True

    return {
        "tracks": tracks,
        "visibility": visibility,
        "point_mask": point_mask,
    }


def map_to_latent_grid(
    tracks: torch.Tensor,
    latent_h: int,
    latent_w: int,
    track_resolution: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[bool]]:
    x = tracks[..., 0]
    y = tracks[..., 1]

    if mode == "auto":
        is_normalized = bool((tracks.max() <= 2.0).item() and (tracks.min() >= -0.5).item())
    elif mode == "normalized":
        is_normalized = True
    elif mode == "pixel":
        is_normalized = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if is_normalized:
        gx = torch.floor(x * max(latent_w - 1, 1))
        gy = torch.floor(y * max(latent_h - 1, 1))
    else:
        src_w = torch.clamp(track_resolution[:, 0].view(-1, 1, 1), min=1.0)
        src_h = torch.clamp(track_resolution[:, 1].view(-1, 1, 1), min=1.0)
        gx = torch.floor(x / src_w * max(latent_w - 1, 1))
        gy = torch.floor(y / src_h * max(latent_h - 1, 1))

    return gx.long(), gy.long(), is_normalized if mode == "auto" else None


def scatter_count_canvas(
    gx: torch.Tensor,
    gy: torch.Tensor,
    visibility: torch.Tensor,
    point_mask: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> Tuple[torch.Tensor, int, int]:
    bsz, t_steps, _ = gx.shape
    canvas = torch.zeros((bsz, t_steps, latent_h, latent_w), dtype=torch.float32)
    vis_bool = visibility > 0.5
    visible_points = int((vis_bool & point_mask[:, None, :]).sum().item())
    visible_inbound_points = 0

    for b in range(bsz):
        valid_points = point_mask[b]
        for t in range(t_steps):
            valid = vis_bool[b, t] & valid_points
            if not torch.any(valid):
                continue
            gx_bt = gx[b, t, valid]
            gy_bt = gy[b, t, valid]
            in_bound = (gx_bt >= 0) & (gx_bt < latent_w) & (gy_bt >= 0) & (gy_bt < latent_h)
            if not torch.any(in_bound):
                continue
            visible_inbound_points += int(in_bound.sum().item())
            gx_bt = gx_bt[in_bound]
            gy_bt = gy_bt[in_bound]
            flat_idx = gy_bt * latent_w + gx_bt
            flat_canvas = canvas[b, t].view(-1)
            flat_canvas.index_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )

    return canvas, visible_points, visible_inbound_points


def build_mode_result(
    mode: str,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    point_mask: torch.Tensor,
    track_resolution: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> ModeResult:
    gx, gy, auto_decision = map_to_latent_grid(
        tracks=tracks,
        latent_h=latent_h,
        latent_w=latent_w,
        track_resolution=track_resolution,
        mode=mode,
    )
    canvas, visible_points, visible_inbound_points = scatter_count_canvas(
        gx=gx,
        gy=gy,
        visibility=visibility,
        point_mask=point_mask,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    nonzero_sites = int((canvas > 0).sum().item())
    total_sites = int(canvas.numel())
    active_cells_union = int((canvas.sum(dim=(0, 1)) > 0).sum().item())
    total_hits = int(canvas.sum().item())
    collision_cells = int((canvas >= 2).sum().item())
    collision_hits = int(torch.clamp(canvas - 1.0, min=0.0).sum().item())
    max_cell_count = int(canvas.max().item()) if total_hits > 0 else 0
    return ModeResult(
        mode=mode,
        decision_is_normalized=auto_decision,
        canvas=canvas,
        nonzero_sites=nonzero_sites,
        total_sites=total_sites,
        nonzero_ratio=float(nonzero_sites / max(total_sites, 1)),
        active_cells_union=active_cells_union,
        visible_points=visible_points,
        visible_inbound_points=visible_inbound_points,
        total_hits=total_hits,
        collision_cells=collision_cells,
        collision_cell_ratio=float(collision_cells / max(nonzero_sites, 1)),
        collision_hits=collision_hits,
        collision_hit_ratio=float(collision_hits / max(total_hits, 1)),
        max_cell_count=max_cell_count,
    )


def render_mode_panels(
    result: ModeResult,
    output_path: str,
    frame_index: int,
    heat_percentile: float,
    heat_gamma: float,
) -> None:
    canvas = result.canvas
    bsz, t_steps, latent_h, latent_w = canvas.shape
    frame_idx = min(max(frame_index, 0), t_steps - 1)

    union_heat = canvas.sum(dim=(0, 1)).cpu().numpy()
    occupancy = (union_heat > 0).astype(np.float32)
    union_overlap = np.maximum(union_heat - 1.0, 0.0)
    union_collision = (union_heat >= 2.0).astype(np.float32)
    frame_heat = canvas[0, frame_idx].cpu().numpy()
    frame_overlap = np.maximum(frame_heat - 1.0, 0.0)
    union_display = _display_heatmap(
        union_heat, percentile=heat_percentile, gamma=heat_gamma
    )
    union_overlap_display = _display_heatmap(
        union_overlap, percentile=heat_percentile, gamma=heat_gamma
    )
    frame_display = _display_heatmap(
        frame_heat, percentile=heat_percentile, gamma=heat_gamma
    )
    frame_overlap_display = _display_heatmap(
        frame_overlap, percentile=heat_percentile, gamma=heat_gamma
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))

    im0 = axes[0, 0].imshow(union_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("union density (sum over batch,time)")
    axes[0, 0].set_xlabel("W_lat")
    axes[0, 0].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[0, 0], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(occupancy, cmap="gray")
    axes[0, 1].set_title("union occupancy (0/1)")
    axes[0, 1].set_xlabel("W_lat")
    axes[0, 1].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[0, 1], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(union_overlap_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("union overlap heat (count-1)")
    axes[0, 2].set_xlabel("W_lat")
    axes[0, 2].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[0, 2], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(frame_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(f"sample0 frame{frame_idx} density")
    axes[1, 0].set_xlabel("W_lat")
    axes[1, 0].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[1, 0], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(frame_overlap_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"sample0 frame{frame_idx} overlap heat (count-1)")
    axes[1, 1].set_xlabel("W_lat")
    axes[1, 1].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[1, 1], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(union_collision, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("union collision mask (count>=2)")
    axes[1, 2].set_xlabel("W_lat")
    axes[1, 2].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[1, 2], latent_h=latent_h, latent_w=latent_w)
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    decision_text = (
        f"auto_is_normalized={result.decision_is_normalized}"
        if result.mode == "auto"
        else "auto_is_normalized=N/A"
    )
    title = (
        f"mode={result.mode} | B={bsz} T={t_steps} "
        f"| active_cells_union={result.active_cells_union} "
        f"| collision_hits_ratio={result.collision_hit_ratio:.4f} "
        f"| collision_cells_ratio={result.collision_cell_ratio:.4f} "
        f"| max_cell_count={result.max_cell_count} "
        f"| {decision_text}"
    )
    subtitle = (
        f"nonzero_ratio={result.nonzero_ratio:.6f} "
        f"| visible_points={result.visible_points} "
        f"| visible_inbound_points={result.visible_inbound_points} "
        f"| collision_hits={result.collision_hits} "
        f"| collision_cells={result.collision_cells}"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=10)
    fig.subplots_adjust(wspace=0.18, right=0.9, top=0.85)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_comparison_grid(
    results: Sequence[ModeResult],
    output_path: str,
    frame_index: int,
    heat_percentile: float,
    heat_gamma: float,
) -> None:
    rows = len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(14.5, 4.2 * rows))
    if rows == 1:
        axes = np.array([axes])  # [1,3]

    for i, result in enumerate(results):
        canvas = result.canvas
        _, t_steps, latent_h, latent_w = canvas.shape
        frame_idx = min(max(frame_index, 0), t_steps - 1)

        union_heat = canvas.sum(dim=(0, 1)).cpu().numpy()
        union_overlap = np.maximum(union_heat - 1.0, 0.0)
        frame_heat = canvas[0, frame_idx].cpu().numpy()
        frame_overlap = np.maximum(frame_heat - 1.0, 0.0)
        union_display = _display_heatmap(
            union_heat, percentile=heat_percentile, gamma=heat_gamma
        )
        union_overlap_display = _display_heatmap(
            union_overlap, percentile=heat_percentile, gamma=heat_gamma
        )
        frame_display = _display_heatmap(
            frame_heat, percentile=heat_percentile, gamma=heat_gamma
        )
        frame_overlap_display = _display_heatmap(
            frame_overlap, percentile=heat_percentile, gamma=heat_gamma
        )

        im0 = axes[i, 0].imshow(union_display, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(
            f"{result.mode}: union brightness-adjusted | cells={result.active_cells_union}"
        )
        axes[i, 0].set_xlabel("W_lat")
        axes[i, 0].set_ylabel("H_lat")
        _apply_latent_axis_style(axes[i, 0], latent_h=latent_h, latent_w=latent_w)
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(union_overlap_display, cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title(
            f"{result.mode}: union overlap heat | hit_ratio={result.collision_hit_ratio:.3f}"
        )
        axes[i, 1].set_xlabel("W_lat")
        axes[i, 1].set_ylabel("H_lat")
        _apply_latent_axis_style(axes[i, 1], latent_h=latent_h, latent_w=latent_w)
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(frame_overlap_display, cmap="magma", vmin=0.0, vmax=1.0)
        decision_text = (
            f"auto={result.decision_is_normalized}" if result.mode == "auto" else ""
        )
        frame_hits = int(frame_heat.sum())
        frame_unique_cells = int((frame_heat > 0).sum())
        frame_overlap_hits = max(frame_hits - frame_unique_cells, 0)
        frame_overlap_ratio = float(frame_overlap_hits / max(frame_hits, 1))
        axes[i, 2].set_title(
            (
                f"{result.mode}: frame{frame_idx} overlap={frame_overlap_ratio:.3f} "
                f"{decision_text}"
            ).strip()
        )
        axes[i, 2].set_xlabel("W_lat")
        axes[i, 2].set_ylabel("H_lat")
        _apply_latent_axis_style(axes[i, 2], latent_h=latent_h, latent_w=latent_w)
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Track canvas mode comparison (with overlap heat)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., :3].copy()


def _ensure_even_hw(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return frame_rgb
    return np.pad(
        frame_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )


def _display_heatmap(
    count_map: np.ndarray,
    percentile: float,
    gamma: float,
) -> np.ndarray:
    display, _, _ = _display_heatmap_with_stats(
        count_map=count_map,
        percentile=percentile,
        gamma=gamma,
    )
    return display


def _display_heatmap_with_stats(
    count_map: np.ndarray,
    percentile: float,
    gamma: float,
) -> Tuple[np.ndarray, float, float]:
    log_map = np.log1p(np.maximum(count_map, 0.0))
    p = float(np.clip(percentile, 50.0, 100.0))
    g = float(max(gamma, 1e-4))
    vmax = float(np.percentile(log_map, p))
    vmax = max(vmax, 1e-8)
    normalized = np.clip(log_map / vmax, 0.0, 1.0)
    return np.power(normalized, g), vmax, g


def _display_heatmap_with_fixed_stats(
    count_map: np.ndarray,
    vmax: float,
    gamma: float,
) -> np.ndarray:
    log_map = np.log1p(np.maximum(count_map, 0.0))
    normalized = np.clip(log_map / max(float(vmax), 1e-8), 0.0, 1.0)
    return np.power(normalized, float(max(gamma, 1e-4)))


def _set_count_ticklabels_for_colorbar(
    cbar,
    vmax: float,
    gamma: float,
    num_ticks: int = 6,
) -> None:
    ticks = np.linspace(0.0, 1.0, int(max(num_ticks, 2)))
    cbar.set_ticks(ticks)
    safe_gamma = float(max(gamma, 1e-8))
    safe_vmax = float(max(vmax, 1e-8))
    clip_count = float(np.expm1(safe_vmax))

    tick_labels: List[str] = []
    for idx, tick in enumerate(ticks):
        if idx == len(ticks) - 1:
            # Values above this level are clipped to 1.0 in display space.
            tick_labels.append(f">={clip_count:.1f}")
            continue
        inv_norm = float(np.clip(tick, 0.0, 1.0)) ** (1.0 / safe_gamma)
        count_val = float(np.expm1(inv_norm * safe_vmax))
        tick_labels.append(f"{count_val:.1f}")
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("count per latent cell")


def render_normalized_canvas_colorbar(
    result: ModeResult,
    output_path: str,
    frame_index: int,
    heat_percentile: float,
    heat_gamma: float,
) -> None:
    canvas = result.canvas
    _, t_steps, latent_h, latent_w = canvas.shape
    frame_idx = min(max(frame_index, 0), t_steps - 1)

    union_heat = canvas.sum(dim=(0, 1)).cpu().numpy()
    frame_heat = canvas[0, frame_idx].cpu().numpy()
    union_display, union_vmax, used_gamma = _display_heatmap_with_stats(
        count_map=union_heat,
        percentile=heat_percentile,
        gamma=heat_gamma,
    )
    frame_display = _display_heatmap_with_fixed_stats(
        count_map=frame_heat,
        vmax=union_vmax,
        gamma=used_gamma,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    im0 = axes[0].imshow(union_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("normalized canvas union heat")
    axes[0].set_xlabel("W_lat")
    axes[0].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[0], latent_h=latent_h, latent_w=latent_w)

    im1 = axes[1].imshow(frame_display, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"normalized canvas frame{frame_idx}")
    axes[1].set_xlabel("W_lat")
    axes[1].set_ylabel("H_lat")
    _apply_latent_axis_style(axes[1], latent_h=latent_h, latent_w=latent_w)

    cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
    _set_count_ticklabels_for_colorbar(
        cbar=cbar,
        vmax=union_vmax,
        gamma=used_gamma,
        num_ticks=6,
    )

    fig.suptitle(
        (
            "Normalized canvas heatmap aligned to one colorbar\n"
            f"(percentile={heat_percentile:.2f}, gamma={heat_gamma:.3f})"
        ),
        fontsize=11,
    )
    fig.subplots_adjust(wspace=0.18, right=0.9, top=0.85)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_normalized_canvas_colorbar_video(
    result: ModeResult,
    output_path: str,
    fps: int,
    max_frames: int,
    heat_percentile: float,
    heat_gamma: float,
) -> None:
    canvas = result.canvas
    _, t_steps, latent_h, latent_w = canvas.shape
    if max_frames > 0:
        t_steps = min(t_steps, int(max_frames))
    if t_steps <= 0:
        return

    union_heat = canvas.sum(dim=(0, 1)).cpu().numpy()
    union_display, union_vmax, used_gamma = _display_heatmap_with_stats(
        count_map=union_heat,
        percentile=heat_percentile,
        gamma=heat_gamma,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=max(1, int(fps)),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame_idx in range(t_steps):
            frame_heat = canvas[0, frame_idx].cpu().numpy()
            frame_display = _display_heatmap_with_fixed_stats(
                count_map=frame_heat,
                vmax=union_vmax,
                gamma=used_gamma,
            )

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
            im0 = axes[0].imshow(union_display, cmap="magma", vmin=0.0, vmax=1.0)
            axes[0].set_title("normalized canvas union heat")
            axes[0].set_xlabel("W_lat")
            axes[0].set_ylabel("H_lat")
            _apply_latent_axis_style(axes[0], latent_h=latent_h, latent_w=latent_w)

            im1 = axes[1].imshow(frame_display, cmap="magma", vmin=0.0, vmax=1.0)
            axes[1].set_title(f"normalized canvas frame{frame_idx}")
            axes[1].set_xlabel("W_lat")
            axes[1].set_ylabel("H_lat")
            _apply_latent_axis_style(axes[1], latent_h=latent_h, latent_w=latent_w)

            cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
            _set_count_ticklabels_for_colorbar(
                cbar=cbar,
                vmax=union_vmax,
                gamma=used_gamma,
                num_ticks=6,
            )

            fig.suptitle(
                (
                    "Normalized canvas heatmap aligned to one colorbar\n"
                    f"(percentile={heat_percentile:.2f}, gamma={heat_gamma:.3f})"
                ),
                fontsize=11,
            )
            fig.subplots_adjust(wspace=0.18, right=0.9, top=0.85)
            frame_rgb = _ensure_even_hw(_fig_to_rgb(fig))
            writer.append_data(frame_rgb)
            plt.close(fig)


def render_track_canvas_video(
    results: Sequence[ModeResult],
    tracks_for_display: torch.Tensor,
    visibility: torch.Tensor,
    point_mask: torch.Tensor,
    output_path: str,
    sample_index: int,
    fps: int,
    max_frames: int,
    display_w: int,
    display_h: int,
    heat_percentile: float,
    heat_gamma: float,
) -> None:
    if len(results) == 0:
        return

    sample_idx = min(max(sample_index, 0), int(tracks_for_display.shape[0]) - 1)
    t_steps = int(min(result.canvas.shape[1] for result in results))
    if max_frames > 0:
        t_steps = min(t_steps, int(max_frames))
    if t_steps <= 0:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=max(1, int(fps)),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame_idx in range(t_steps):
            cols = 1 + len(results)
            fig, axes = plt.subplots(1, cols, figsize=(4.8 * cols, 4.5))
            if cols == 1:
                axes = np.array([axes])

            x = tracks_for_display[sample_idx, frame_idx, :, 0].cpu().numpy()
            y = tracks_for_display[sample_idx, frame_idx, :, 1].cpu().numpy()
            vis = (visibility[sample_idx, frame_idx] > 0.5).cpu().numpy()
            pm = point_mask[sample_idx].cpu().numpy()
            valid = vis & pm

            axes[0].set_facecolor("black")
            axes[0].scatter(x[valid], y[valid], s=5, c="#00e5ff", alpha=0.85, edgecolors="none")
            _apply_pixel_axis_style(axes[0], display_h=display_h, display_w=display_w)
            axes[0].set_title(
                f"track points (sample={sample_idx}, frame={frame_idx})\n"
                f"visible={int(valid.sum())}/{int(pm.sum())}"
            )
            axes[0].set_xlabel("x (source pixels)")
            axes[0].set_ylabel("y (source pixels)")

            for col_idx, result in enumerate(results, start=1):
                heat = result.canvas[sample_idx, frame_idx].cpu().numpy()
                latent_h, latent_w = int(heat.shape[0]), int(heat.shape[1])
                heat_display = _display_heatmap(
                    heat, percentile=heat_percentile, gamma=heat_gamma
                )
                im = axes[col_idx].imshow(
                    heat_display, cmap="magma", vmin=0.0, vmax=1.0
                )
                decision_text = (
                    f"auto={result.decision_is_normalized}" if result.mode == "auto" else ""
                )
                frame_hits = int(heat.sum())
                frame_unique_cells = int((heat > 0).sum())
                frame_overlap_hits = max(frame_hits - frame_unique_cells, 0)
                frame_overlap_ratio = float(frame_overlap_hits / max(frame_hits, 1))
                axes[col_idx].set_title(
                    (
                        f"{result.mode} canvas {decision_text}\n"
                        f"overlap={frame_overlap_ratio:.3f} ({frame_overlap_hits}/{max(frame_hits, 1)})"
                    ).strip()
                )
                axes[col_idx].set_xlabel("W_lat")
                axes[col_idx].set_ylabel("H_lat")
                _apply_latent_axis_style(axes[col_idx], latent_h=latent_h, latent_w=latent_w)
                fig.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)

            fig.suptitle("Frame-wise track and canvas comparison", fontsize=12)
            fig.tight_layout()
            frame_rgb = _ensure_even_hw(_fig_to_rgb(fig))
            writer.append_data(frame_rgb)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.track_npz is not None:
        track_paths = [os.path.abspath(path) for path in args.track_npz]
    else:
        track_paths = resolve_meta_track_paths(
            meta_json=os.path.abspath(args.meta_json),
            data_root=os.path.abspath(args.data_root) if args.data_root else None,
            track_key=args.track_key,
            offset=max(args.meta_offset, 0),
            count=max(args.meta_count, 1),
        )

    items = [load_track_npz(path) for path in track_paths]
    batch = pad_track_batch(items)
    tracks = batch["tracks"]
    visibility = batch["visibility"]
    point_mask = batch["point_mask"]

    if args.apply_track_normalize:
        tracks = tracks.clone()
        tracks[..., 0] = tracks[..., 0] / float(max(args.track_resolution_w, 1))
        tracks[..., 1] = tracks[..., 1] / float(max(args.track_resolution_h, 1))
        tracks_for_display = tracks.clone()
        tracks_for_display[..., 0] = tracks_for_display[..., 0] * float(
            max(args.track_resolution_w, 1)
        )
        tracks_for_display[..., 1] = tracks_for_display[..., 1] * float(
            max(args.track_resolution_h, 1)
        )
    else:
        tracks_for_display = tracks.clone()

    bsz = tracks.shape[0]
    track_resolution = torch.tensor(
        [[float(max(args.track_resolution_w, 1)), float(max(args.track_resolution_h, 1))]]
        * bsz,
        dtype=torch.float32,
    )

    results: List[ModeResult] = []
    normalized_result: Optional[ModeResult] = None
    normalized_colorbar_img: Optional[str] = None
    normalized_colorbar_mp4: Optional[str] = None
    for mode in args.modes:
        result = build_mode_result(
            mode=mode,
            tracks=tracks,
            visibility=visibility,
            point_mask=point_mask,
            track_resolution=track_resolution,
            latent_h=args.latent_h,
            latent_w=args.latent_w,
        )
        results.append(result)

        out_img = os.path.join(args.output_dir, f"canvas_{mode}.png")
        render_mode_panels(
            result=result,
            output_path=out_img,
            frame_index=args.frame_index,
            heat_percentile=args.heat_percentile,
            heat_gamma=args.heat_gamma,
        )
        if mode == "normalized":
            normalized_colorbar_img = os.path.join(
                args.output_dir, "canvas_normalized_colorbar.png"
            )
            render_normalized_canvas_colorbar(
                result=result,
                output_path=normalized_colorbar_img,
                frame_index=args.frame_index,
                heat_percentile=args.heat_percentile,
                heat_gamma=args.heat_gamma,
            )
            normalized_result = result

    compare_img = os.path.join(args.output_dir, "canvas_compare.png")
    render_comparison_grid(
        results=results,
        output_path=compare_img,
        frame_index=args.frame_index,
        heat_percentile=args.heat_percentile,
        heat_gamma=args.heat_gamma,
    )
    compare_mp4 = os.path.join(args.output_dir, "canvas_compare.mp4")
    if args.save_mp4:
        render_track_canvas_video(
            results=results,
            tracks_for_display=tracks_for_display,
            visibility=visibility,
            point_mask=point_mask,
            output_path=compare_mp4,
            sample_index=args.sample_index,
            fps=args.fps,
            max_frames=args.max_frames,
            display_w=args.track_resolution_w,
            display_h=args.track_resolution_h,
            heat_percentile=args.heat_percentile,
            heat_gamma=args.heat_gamma,
        )
        if normalized_result is not None:
            normalized_colorbar_mp4 = os.path.join(
                args.output_dir, "canvas_normalized_colorbar.mp4"
            )
            render_normalized_canvas_colorbar_video(
                result=normalized_result,
                output_path=normalized_colorbar_mp4,
                fps=args.fps,
                max_frames=args.max_frames,
                heat_percentile=args.heat_percentile,
                heat_gamma=args.heat_gamma,
            )

    print("=== Track canvas debug summary ===")
    print(f"loaded_tracks={len(track_paths)}")
    print(f"apply_track_normalize={args.apply_track_normalize}")
    print(f"latent_grid=({args.latent_h}, {args.latent_w})")
    print(f"track_resolution=({args.track_resolution_h}, {args.track_resolution_w})")
    print(
        f"heat_display=(percentile={args.heat_percentile:.2f}, gamma={args.heat_gamma:.3f})"
    )
    print(f"output_dir={os.path.abspath(args.output_dir)}")
    print(f"comparison_plot={compare_img}")
    if normalized_colorbar_img is not None:
        print(f"normalized_colorbar_plot={normalized_colorbar_img}")
    if args.save_mp4:
        print(f"comparison_video={compare_mp4}")
    if normalized_colorbar_mp4 is not None:
        print(f"normalized_colorbar_video={normalized_colorbar_mp4}")
    for result in results:
        decision_text = (
            f"auto_is_normalized={result.decision_is_normalized}"
            if result.mode == "auto"
            else "auto_is_normalized=N/A"
        )
        print(
            f"[{result.mode}] {decision_text} "
            f"nonzero_ratio={result.nonzero_ratio:.6f} "
            f"active_cells_union={result.active_cells_union} "
            f"visible_points={result.visible_points} "
            f"visible_inbound_points={result.visible_inbound_points} "
            f"collision_hit_ratio={result.collision_hit_ratio:.6f} "
            f"collision_cell_ratio={result.collision_cell_ratio:.6f} "
            f"collision_hits={result.collision_hits} "
            f"collision_cells={result.collision_cells} "
            f"max_cell_count={result.max_cell_count}"
        )


if __name__ == "__main__":
    main()
