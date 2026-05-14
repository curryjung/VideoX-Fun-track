from __future__ import annotations

from typing import Any

import numpy as np
import torch


_LPIPS_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"PSNR shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim == 4:
        return float(np.mean([compute_psnr(g, p) for g, p in zip(gt, pred)]))

    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def _infer_image_data_range(gt: np.ndarray, pred: np.ndarray) -> float:
    if np.issubdtype(gt.dtype, np.integer) or np.issubdtype(pred.dtype, np.integer):
        return 255.0
    max_value = max(float(np.nanmax(gt)), float(np.nanmax(pred)))
    min_value = min(float(np.nanmin(gt)), float(np.nanmin(pred)))
    if min_value >= 0.0 and max_value <= 1.0:
        return 1.0
    if min_value >= -1.0 and max_value <= 1.0:
        return 2.0
    return max(max_value - min_value, 1.0)


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"SSIM shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim == 4:
        return float(np.mean([compute_ssim(g, p) for g, p in zip(gt, pred)]))
    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(f"SSIM expects frames with shape (H, W, 3), got {gt.shape}")

    return float(
        structural_similarity(
            gt,
            pred,
            channel_axis=-1,
            data_range=_infer_image_data_range(gt, pred),
        )
    )


def compute_epe(
    input_tracks: np.ndarray,
    input_visibility: np.ndarray,
    generated_tracks: np.ndarray,
) -> float:
    input_tracks = np.asarray(input_tracks, dtype=np.float32)
    input_visibility = np.asarray(input_visibility, dtype=np.float32)
    generated_tracks = np.asarray(generated_tracks, dtype=np.float32)
    if input_tracks.ndim != 3 or generated_tracks.ndim != 3:
        raise ValueError("EPE expects track arrays with shape (T, N, 2).")
    if input_visibility.ndim != 2:
        raise ValueError("EPE expects visibility with shape (T, N).")

    frames = min(input_tracks.shape[0], generated_tracks.shape[0])
    points = min(input_tracks.shape[1], generated_tracks.shape[1])
    visible = input_visibility[:frames, :points].astype(bool)
    diff = input_tracks[:frames, :points] - generated_tracks[:frames, :points]
    distances = np.linalg.norm(diff, axis=-1)
    values = distances[visible]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def _frames_to_lpips_tensor(frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"LPIPS expects frames with shape (T, H, W, 3), got {frames.shape}")

    tensor = torch.from_numpy(frames.astype(np.float32, copy=False)).permute(0, 3, 1, 2)
    if float(tensor.max()) > 1.5:
        tensor = tensor / 127.5 - 1.0
    else:
        tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def build_lpips_model(net: str = "alex", device: torch.device | str | None = None) -> Any:
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS metric requires the `lpips` package. Install it with `pip install lpips` "
            "or reinstall this project from its updated requirements."
        ) from exc

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(net), str(dev))
    if key not in _LPIPS_MODEL_CACHE:
        model = lpips.LPIPS(net=net, verbose=False).to(dev).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_MODEL_CACHE[key] = model
    return _LPIPS_MODEL_CACHE[key]


def compute_lpips(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    model: Any | None = None,
    net: str = "alex",
    device: torch.device | str | None = None,
    batch_size: int = 16,
) -> float:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"LPIPS shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim == 3:
        gt = gt[None]
        pred = pred[None]
    if gt.ndim != 4:
        raise ValueError(f"LPIPS expects frames with shape (T, H, W, 3), got {gt.shape}")
    if gt.shape[0] == 0:
        return float("nan")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lpips_model = model if model is not None else build_lpips_model(net=net, device=dev)
    batch_size = max(1, int(batch_size))
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, gt.shape[0], batch_size):
            end = min(start + batch_size, gt.shape[0])
            gt_batch = _frames_to_lpips_tensor(gt[start:end], dev)
            pred_batch = _frames_to_lpips_tensor(pred[start:end], dev)
            values.append(lpips_model(gt_batch, pred_batch).reshape(-1).detach().cpu())
    return float(torch.cat(values).mean().item())


def compute_metric(
    name: str,
    *,
    gt_frames: np.ndarray,
    generated_frames: np.ndarray,
    input_tracks: np.ndarray,
    input_visibility: np.ndarray,
    generated_tracks: np.ndarray,
    lpips_model: Any | None = None,
    lpips_net: str = "alex",
    lpips_device: torch.device | str | None = None,
    lpips_batch_size: int = 16,
) -> float:
    metric = name.strip().lower()
    if metric == "psnr":
        return compute_psnr(gt_frames, generated_frames)
    if metric == "ssim":
        return compute_ssim(gt_frames, generated_frames)
    if metric == "epe":
        return compute_epe(input_tracks, input_visibility, generated_tracks)
    if metric == "lpips":
        return compute_lpips(
            gt_frames,
            generated_frames,
            model=lpips_model,
            net=lpips_net,
            device=lpips_device,
            batch_size=lpips_batch_size,
        )
    raise ValueError(f"Unsupported metric: {name}")
