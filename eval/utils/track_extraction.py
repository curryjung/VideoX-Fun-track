"""
CoTracker3-based track extraction for evaluation.

Extracts point tracks from a video using a uniform 50x50 grid on the first frame,
matching the protocol in the Self Forcing paper (Section 4.1 / Appendix E).
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

COTRACKER_ROOT = "/data/project-vilab/jaeseok/co-tracker"

# Common checkpoint search paths (offline / CoTracker3 variants).
_CHECKPOINT_SEARCH_PATHS = [
    os.path.join(COTRACKER_ROOT, "checkpoints", "scaled_offline.pth"),
    os.path.join(COTRACKER_ROOT, "checkpoints", "cotracker3_offline.pth"),
    os.path.join(COTRACKER_ROOT, "checkpoints", "cotracker2.pth"),
]


def _find_checkpoint(user_path: str | None) -> str | None:
    if user_path and os.path.isfile(user_path):
        return user_path
    for p in _CHECKPOINT_SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    return None


def load_cotracker(checkpoint: str | None = None, device: torch.device | None = None):
    """
    Load CoTracker3 (offline) and return the model.

    Call this once and pass the returned model to extract_tracks_at_queries()
    to avoid reloading on every video.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if COTRACKER_ROOT not in sys.path:
        sys.path.insert(0, COTRACKER_ROOT)

    from cotracker.predictor import CoTrackerPredictor

    ckpt = _find_checkpoint(checkpoint)
    if ckpt is None:
        print("[track_extraction] No local checkpoint found; loading via torch.hub (CoTracker3 offline).")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    else:
        print(f"[track_extraction] Loading CoTracker from: {ckpt}")
        model = CoTrackerPredictor(checkpoint=ckpt, offline=True)

    return model.to(dev).eval()


def load_video_frames(video_path: str, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load video as uint8 numpy array of shape (T, H, W, 3).

    Supports:
      - .mp4 / .avi / other video files (via cv2)
      - directory of frame images (sorted by filename)
    """
    import cv2

    p = Path(video_path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        frame_paths = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in frame_paths]
    else:
        cap = cv2.VideoCapture(str(p))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from: {video_path}")

    if target_hw is not None:
        h, w = target_hw
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]

    return np.stack(frames, axis=0)  # (T, H, W, 3)


def extract_tracks_at_queries(
    video_frames: np.ndarray,
    query_points: np.ndarray,
    model=None,
    checkpoint: str | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract tracks starting from specific query points on frame 0.

    Args:
        video_frames: (T, H, W, 3) uint8 numpy array.
        query_points: (N, 2) float32 – pixel coordinates (x, y) on frame 0.
        model: pre-loaded CoTracker model (from load_cotracker()). Loaded here
               from checkpoint/torch.hub if None — prefer passing a pre-loaded
               model to avoid reloading on every call.
        checkpoint: path to CoTracker checkpoint (used only when model is None).
        device: torch device string (used only when model is None).

    Returns:
        tracks: (T, N, 2) float32 – pixel coordinates (x, y).
        visibility: (T, N) float32 – 1 = visible, 0 = occluded.
    """
    if model is None:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = load_cotracker(checkpoint, dev)

    dev = next(model.parameters()).device

    video_t = (
        torch.from_numpy(video_frames)
        .permute(0, 3, 1, 2)  # (T, 3, H, W)
        .unsqueeze(0)          # (1, T, 3, H, W)
        .float()
        .to(dev)
    )

    N = query_points.shape[0]
    # CoTracker queries: (B, N, 3) as (t, x, y) with t=0 for first-frame queries
    queries_t = torch.zeros((1, N, 3), dtype=torch.float32, device=dev)
    queries_t[0, :, 1] = torch.from_numpy(query_points[:, 0])  # x
    queries_t[0, :, 2] = torch.from_numpy(query_points[:, 1])  # y

    with torch.no_grad():
        tracks, visibility = model(video_t, queries=queries_t)

    # tracks: (B, T, N, 2) → (T, N, 2); visibility: (B, T, N) → (T, N)
    tracks = tracks[0].cpu().numpy().astype(np.float32)
    visibility = visibility[0].cpu().numpy().astype(np.float32)
    return tracks, visibility


def build_uniform_grid_queries(H: int, W: int, grid_size: int = 50) -> np.ndarray:
    """
    Build 50x50 uniform grid query points on the first frame.

    Returns:
        queries: (N, 2) float32 – pixel coordinates (x, y).
    """
    xs = np.linspace(0, W - 1, grid_size, dtype=np.float32)
    ys = np.linspace(0, H - 1, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    queries = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (N, 2) as (x, y)
    return queries


def save_tracks_npz(
    path: str,
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> None:
    """Save tracks to .npz matching the project's existing format."""
    np.savez_compressed(path, tracks=tracks, visibility=visibility)


def load_tracks_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load tracks from .npz file."""
    data = np.load(path)
    if "tracks_compressed" in data:
        tracks = data["tracks_compressed"]
    else:
        tracks = data["tracks"]

    if "visibility_compressed" in data:
        visibility = data["visibility_compressed"]
    elif "visibility" in data:
        visibility = data["visibility"]
    else:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)

    # Normalise shape to (T, N, 2) and (T, N)
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)

    return tracks.astype(np.float32), visibility.astype(np.float32)
