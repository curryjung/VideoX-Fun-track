"""
Fréchet Video Distance (FVD).

Uses the I3D network pretrained on Kinetics-400 to extract video features,
then computes the Fréchet distance between GT and generated feature distributions.

Requires:
  pip install torch torchvision

Optional (for faster I3D loading):
  pip install tensorflow  # for the original TF checkpoint (not required here)

Reference:
  Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges"
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def _load_i3d(device: torch.device):
    """
    Load I3D (RGB) from torch.hub (facebookresearch/pytorchvideo or similar).

    Falls back to a minimal stub that returns random features if the model
    cannot be loaded, so the rest of the pipeline still runs.
    """
    try:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "i3d_r50",
            pretrained=True,
        )
        model = model.to(device).eval()
        # Remove the final classification head; keep feature trunk.
        model.blocks = model.blocks[:-1]
        print("[fvd] I3D loaded via torch.hub.")
        return model
    except Exception as e:
        print(f"[fvd] Could not load I3D ({e}). FVD will return NaN.")
        return None


def _extract_features(model, video_clips: list[np.ndarray], device: torch.device) -> np.ndarray:
    """
    Extract I3D features for a list of video clips.

    Args:
        video_clips: list of (T, H, W, 3) uint8 arrays.
        device: torch device.

    Returns:
        features: (N, D) float32.
    """
    feats = []
    with torch.no_grad():
        for clip in video_clips:
            # I3D expects (B, C, T, H, W) float in [0, 1], resized to 224x224
            t = (
                torch.from_numpy(clip)
                .permute(0, 3, 1, 2)  # (T, 3, H, W)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(device)
            )
            t = F.interpolate(t[0], size=(224, 224), mode="bilinear", align_corners=False)
            t = t.unsqueeze(0)  # (1, 3, T, 224, 224)  — wait, need (B, C, T, H, W)
            # Actually reshape to (B, C, T, H, W)
            t = t.permute(0, 2, 1, 3, 4)  # already (B, T, C, H, W)? no…
            # Correct: input is (B, C, T, H, W)
            # After unsqueeze: (1, T, 3, H, W) → permute to (1, 3, T, H, W)
            t = t.permute(0, 2, 1, 3, 4)
            feat = model(t)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            feats.append(feat.squeeze().cpu().numpy())
    return np.stack(feats, axis=0)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fvd(
    gt_clips: list[np.ndarray],
    gen_clips: list[np.ndarray],
    device: Optional[torch.device] = None,
    i3d_model=None,
) -> float:
    """
    Compute FVD between a set of GT and generated video clips.

    Args:
        gt_clips: list of (T, H, W, 3) uint8 arrays.
        gen_clips: list of (T, H, W, 3) uint8 arrays (same length as gt_clips).
        device: torch device (defaults to cuda if available).
        i3d_model: pre-loaded I3D model (loaded via load_i3d()); loaded here if None.

    Returns:
        FVD score (lower = better). NaN if I3D could not be loaded.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if i3d_model is None:
        i3d_model = _load_i3d(dev)
    if i3d_model is None:
        return float("nan")

    gt_feats = _extract_features(i3d_model, gt_clips, dev)   # (N, D)
    gen_feats = _extract_features(i3d_model, gen_clips, dev)  # (N, D)

    mu_gt, sigma_gt = gt_feats.mean(0), np.cov(gt_feats, rowvar=False)
    mu_gen, sigma_gen = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)

    return _frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)


def load_i3d(device: Optional[torch.device] = None):
    """Load and return the I3D model for reuse across calls."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _load_i3d(dev)
