from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrackSampleConfig:
    max_points: int = -1
    sample_mode: str = "uniform"
    sort_selected_indices: bool = True
    seed: Optional[int] = None
    point_id_mode: str = "original"


def select_track_indices_uniform(num_points: int, max_points: int) -> np.ndarray:
    """Deterministic point sampling that preserves coverage over source order."""
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


def select_track_indices_random(
    num_points: int,
    max_points: int,
    *,
    sort_selected_indices: bool,
    seed: Optional[int],
) -> np.ndarray:
    if max_points <= 0 or max_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_points)[:max_points]
    if sort_selected_indices:
        idx = np.sort(idx)
    return idx.astype(np.int64, copy=False)


def select_track_indices(
    num_points: int,
    max_points: int,
    *,
    sample_mode: str,
    sort_selected_indices: bool,
    seed: Optional[int],
) -> np.ndarray:
    mode = str(sample_mode).strip().lower()
    if mode == "uniform":
        return select_track_indices_uniform(num_points, max_points)
    if mode == "random":
        return select_track_indices_random(
            num_points,
            max_points,
            sort_selected_indices=sort_selected_indices,
            seed=seed,
        )
    raise ValueError(f"Unsupported track point sample mode: {sample_mode}")


def normalize_track_arrays(
    tracks: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Expected tracks with shape (T, N, 2), got {tracks.shape}")

    if visibility is None:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)
    else:
        visibility = np.asarray(visibility, dtype=np.float32)
        if visibility.ndim == 3 and visibility.shape[0] == 1:
            visibility = visibility[0]
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility.squeeze(-1)

    if visibility.ndim != 2:
        raise ValueError(f"Expected visibility with shape (T, N), got {visibility.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"Track/visibility shape mismatch: tracks={tracks.shape}, "
            f"visibility={visibility.shape}"
        )
    return tracks.astype(np.float32, copy=False), visibility.astype(np.float32, copy=False)


def load_tracks_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    elif "tracks" in data:
        tracks = data["tracks"]
    else:
        raise KeyError(f"Track file missing `tracks` key: {path}")

    if "visibility_compressed" in data:
        visibility = data["visibility_compressed"]
    elif "visibility" in data:
        visibility = data["visibility"]
    else:
        visibility = None
    return normalize_track_arrays(tracks, visibility)


def save_tracks_npz(
    path: str | Path,
    tracks: np.ndarray,
    visibility: np.ndarray,
    **extra_arrays: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks, visibility = normalize_track_arrays(tracks, visibility)
    payload = {"tracks": tracks, "visibility": visibility}
    payload.update({k: v for k, v in extra_arrays.items() if v is not None})
    np.savez_compressed(path, **payload)


def sample_track_subset(
    tracks: np.ndarray,
    visibility: np.ndarray,
    config: TrackSampleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks, visibility = normalize_track_arrays(tracks, visibility)
    point_id_mode = str(config.point_id_mode).strip().lower()
    if point_id_mode not in {"original", "local"}:
        raise ValueError(f"Unsupported point_id_mode: {config.point_id_mode}")

    keep_idx = select_track_indices(
        tracks.shape[1],
        int(config.max_points),
        sample_mode=config.sample_mode,
        sort_selected_indices=bool(config.sort_selected_indices),
        seed=config.seed,
    )
    subset_tracks = tracks[:, keep_idx]
    subset_visibility = visibility[:, keep_idx]
    if point_id_mode == "local":
        point_ids = np.arange(keep_idx.size, dtype=np.int64)
    else:
        point_ids = keep_idx.astype(np.int64, copy=False)
    return subset_tracks, subset_visibility, keep_idx, point_ids
