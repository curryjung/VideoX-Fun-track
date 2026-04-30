"""
Visual fidelity metrics: PSNR, SSIM, LPIPS.

All functions operate on uint8 numpy frames (H, W, 3) or stacks (T, H, W, 3).
"""

import numpy as np
import torch
from typing import Optional


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    PSNR between two (H, W, 3) uint8 frames.

    If arrays are (T, H, W, 3), returns the mean over frames.
    """
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    SSIM between two (H, W, 3) uint8 frames.

    If arrays are (T, H, W, 3), returns the mean over frames.
    """
    from skimage.metrics import structural_similarity

    if gt.ndim == 4:
        return float(np.mean([
            structural_similarity(g, p, channel_axis=-1, data_range=255)
            for g, p in zip(gt, pred)
        ]))
    return float(structural_similarity(gt, pred, channel_axis=-1, data_range=255))


def load_lpips(device: torch.device):
    """
    Load the LPIPS (AlexNet) network. Returns None if lpips is not installed.
    """
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").to(device)
        fn.eval()
        return fn
    except ImportError:
        print("[visual_fidelity] lpips not installed — LPIPS will be NaN. Run: pip install lpips")
        return None


def compute_lpips(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    lpips_fn,
    device: torch.device,
) -> float:
    """
    Mean LPIPS over all frames.

    Args:
        gt_frames: (T, H, W, 3) uint8.
        pred_frames: (T, H, W, 3) uint8.
        lpips_fn: result of load_lpips(); None → returns NaN.
        device: torch device.

    Returns:
        mean LPIPS score (lower = better).
    """
    if lpips_fn is None:
        return float("nan")

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        # (T, H, W, 3) → (T, 3, H, W), normalised to [-1, 1]
        return (
            torch.from_numpy(arr)
            .permute(0, 3, 1, 2)
            .float()
            .div(127.5)
            .sub(1.0)
            .to(device)
        )

    gt_t = to_tensor(gt_frames)
    pred_t = to_tensor(pred_frames)
    vals = []
    with torch.no_grad():
        for g, p in zip(gt_t, pred_t):
            vals.append(lpips_fn(g.unsqueeze(0), p.unsqueeze(0)).item())
    return float(np.mean(vals))


def compute_all(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    lpips_fn=None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute PSNR, SSIM, and LPIPS for a pair of (T, H, W, 3) uint8 video clips.

    Returns:
        dict with keys: psnr, ssim, lpips (per-video means).
    """
    psnr_vals = [compute_psnr(g, p) for g, p in zip(gt_frames, pred_frames)]
    ssim_vals = [compute_ssim(g, p) for g, p in zip(gt_frames, pred_frames)]

    dev = device or torch.device("cpu")
    lpips_val = compute_lpips(gt_frames, pred_frames, lpips_fn, dev)

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": lpips_val,
    }
