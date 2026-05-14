from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


DEFAULT_COTRACKER_ROOT = Path("/data/project-vilab/jaeseok/co-tracker")
DEFAULT_CHECKPOINTS = (
    DEFAULT_COTRACKER_ROOT / "checkpoints" / "scaled_offline.pth",
    DEFAULT_COTRACKER_ROOT / "checkpoints" / "cotracker3_offline.pth",
    DEFAULT_COTRACKER_ROOT / "checkpoints" / "cotracker2.pth",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_davis_videos(davis_root: str | Path) -> list[Path]:
    root = Path(davis_root)
    if not root.exists():
        raise FileNotFoundError(f"DAVIS root not found: {root}")

    videos: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
            videos.append(path)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    return videos


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames],
        axis=0,
    )


def load_video_frames(
    video_path: str | Path,
    target_hw: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    path = Path(video_path)
    frames: list[np.ndarray] = []
    if path.is_dir():
        frame_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from: {path}")

    arr = np.stack(frames, axis=0)
    if target_hw is not None:
        arr = resize_frames(arr, height=int(target_hw[0]), width=int(target_hw[1]))
    return arr.astype(np.uint8, copy=False)


def subsample_frames(frames: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    total = int(frames.shape[0])
    target_len = int(target_len)
    if target_len <= 0 or total <= target_len:
        return frames, np.arange(total, dtype=np.int64)
    if target_len == 1:
        return frames[:1], np.array([0], dtype=np.int64)

    middle = np.linspace(1, total - 2, target_len - 2, dtype=np.float64)
    indices = np.concatenate([[0], np.round(middle).astype(np.int64), [total - 1]])
    indices = np.unique(np.clip(indices, 0, total - 1))
    if indices.size < target_len:
        fallback = np.round(np.linspace(0, total - 1, target_len)).astype(np.int64)
        indices = np.unique(fallback)
    return frames[indices], indices.astype(np.int64, copy=False)


def prepare_davis_frames(
    frames: np.ndarray,
    model_video_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frames.shape[0] > int(model_video_length):
        return subsample_frames(frames, int(model_video_length))
    return frames, np.arange(frames.shape[0], dtype=np.int64)


def build_uniform_grid_queries(height: int, width: int, grid_size: int = 50) -> np.ndarray:
    xs = np.linspace(0, width - 1, int(grid_size), dtype=np.float32)
    ys = np.linspace(0, height - 1, int(grid_size), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)


def _find_checkpoint(checkpoint: Optional[str]) -> Optional[str]:
    if checkpoint:
        path = Path(checkpoint)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"CoTracker checkpoint not found: {path}")
    for path in DEFAULT_CHECKPOINTS:
        if path.is_file():
            return str(path)
    return None


def load_cotracker(
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from cotracker.predictor import CoTrackerPredictor

    ckpt = _find_checkpoint(checkpoint)
    if ckpt is None:
        print("[cotracker] No local checkpoint found; loading torch.hub CoTracker3 offline.")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    else:
        print(f"[cotracker] Loading checkpoint: {ckpt}")
        model = CoTrackerPredictor(checkpoint=ckpt, offline=True)
    return model.to(dev).eval()


def extract_tracks_at_queries(
    video_frames: np.ndarray,
    query_points: np.ndarray,
    *,
    model,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
        raise ValueError(f"Expected video frames (T, H, W, 3), got {video_frames.shape}")
    query_points = np.asarray(query_points, dtype=np.float32)
    if query_points.ndim != 2 or query_points.shape[-1] != 2:
        raise ValueError(f"Expected query points (N, 2), got {query_points.shape}")

    dev = device or next(model.parameters()).device
    video_t = (
        torch.from_numpy(video_frames)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .float()
        .to(dev)
    )
    queries_t = torch.zeros((1, query_points.shape[0], 3), dtype=torch.float32, device=dev)
    queries_t[0, :, 1:] = torch.as_tensor(query_points, dtype=torch.float32, device=dev)

    with torch.no_grad():
        tracks, visibility = model(video_t, queries=queries_t)

    return (
        tracks[0].detach().cpu().numpy().astype(np.float32),
        visibility[0].detach().cpu().numpy().astype(np.float32),
    )
