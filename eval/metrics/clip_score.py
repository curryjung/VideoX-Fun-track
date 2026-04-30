"""
CLIP Score for text-video alignment.

Computes cosine similarity between CLIP text embeddings and CLIP frame embeddings,
averaged over sampled frames of the generated video.

Requires:
  pip install transformers

Reference:
  Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: Optional[torch.device] = None):
    """
    Load CLIP model and processor.

    Returns:
        (model, processor) tuple, or (None, None) if transformers not available.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).to(dev).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[clip_score] CLIP loaded: {model_name}")
        return model, processor
    except Exception as e:
        print(f"[clip_score] Could not load CLIP ({e}). CLIP Score will return NaN.")
        return None, None


def compute_clip_score(
    video_frames: np.ndarray,
    prompt: str,
    clip_model=None,
    clip_processor=None,
    device: Optional[torch.device] = None,
    num_frames: int = 8,
) -> float:
    """
    Mean CLIP cosine similarity between a text prompt and uniformly sampled frames.

    Args:
        video_frames: (T, H, W, 3) uint8 numpy array.
        prompt: text description of the video.
        clip_model: pre-loaded CLIP model (via load_clip()).
        clip_processor: pre-loaded CLIP processor.
        device: torch device.
        num_frames: number of frames to sample for efficiency.

    Returns:
        Mean CLIP score in [0, 1] (higher = better). NaN if CLIP not available.
    """
    if clip_model is None or clip_processor is None:
        return float("nan")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = video_frames.shape[0]

    # Uniformly sample frames
    indices = np.linspace(0, T - 1, min(num_frames, T), dtype=int)
    sampled = [video_frames[i] for i in indices]  # list of (H, W, 3) uint8

    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in sampled]

    inputs = clip_processor(
        text=[prompt],
        images=pil_frames,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        text_emb = F.normalize(outputs.text_embeds, dim=-1)   # (1, D)
        img_emb = F.normalize(outputs.image_embeds, dim=-1)   # (N, D)

    scores = (img_emb @ text_emb.T).squeeze(-1)  # (N,)
    return float(scores.mean().item())


def compute_clip_score_batch(
    video_clips: list[np.ndarray],
    prompts: list[str],
    clip_model=None,
    clip_processor=None,
    device: Optional[torch.device] = None,
    num_frames: int = 8,
) -> list[float]:
    """
    Compute CLIP scores for a batch of (video, prompt) pairs.

    Returns:
        List of per-video CLIP scores.
    """
    return [
        compute_clip_score(v, p, clip_model, clip_processor, device, num_frames)
        for v, p in zip(video_clips, prompts)
    ]
