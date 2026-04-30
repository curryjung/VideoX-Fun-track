"""
End-Point Error (EPE) for motion transfer evaluation.

Protocol (Section 4.2):
  - Query CoTracker3 on the generated video at the same grid points used as
    input conditioning.
  - Compute L2 distance between visible input tracks and re-extracted tracks.
  - Average over all visible (t, i) pairs.
"""

import numpy as np


def compute_epe(
    input_tracks: np.ndarray,
    input_visibility: np.ndarray,
    gen_tracks: np.ndarray,
) -> float:
    """
    End-Point Error between visible input (GT) tracks and tracks re-extracted
    from the generated video.

    Args:
        input_tracks: (T, N, 2) float32 – pixel coords (x, y) from GT video.
        input_visibility: (T, N) float32 – 1 = visible in GT video.
        gen_tracks: (T, N, 2) float32 – pixel coords re-extracted from generated video.

    Returns:
        Mean EPE in pixels over all visible (t, point) pairs.
        Returns NaN if no visible points exist.
    """
    T = min(input_tracks.shape[0], gen_tracks.shape[0])
    vis = input_visibility[:T].astype(bool)           # (T, N)
    diff = input_tracks[:T] - gen_tracks[:T]          # (T, N, 2)
    epe_map = np.linalg.norm(diff, axis=-1)           # (T, N)
    visible_epe = epe_map[vis]
    if visible_epe.size == 0:
        return float("nan")
    return float(visible_epe.mean())


def compute_epe_per_frame(
    input_tracks: np.ndarray,
    input_visibility: np.ndarray,
    gen_tracks: np.ndarray,
) -> np.ndarray:
    """
    Per-frame EPE averaged over visible points in each frame.

    Returns:
        epe_per_frame: (T,) float32, NaN for frames with no visible points.
    """
    T = min(input_tracks.shape[0], gen_tracks.shape[0])
    vis = input_visibility[:T].astype(bool)
    diff = input_tracks[:T] - gen_tracks[:T]
    epe_map = np.linalg.norm(diff, axis=-1)  # (T, N)

    per_frame = np.full(T, float("nan"), dtype=np.float32)
    for t in range(T):
        vis_t = vis[t]
        if vis_t.any():
            per_frame[t] = float(epe_map[t][vis_t].mean())
    return per_frame
