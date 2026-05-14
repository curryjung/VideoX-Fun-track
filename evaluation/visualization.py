from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _point_color(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(index)
    color = rng.integers(64, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def save_track_overlay_video(
    frames_rgb: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_path: str | Path,
    *,
    fps: int = 16,
    trace_frames: int = 8,
    radius: int = 2,
    line_width: int = 1,
    scale: float = 0.5,
    crf: int | None = 32,
) -> None:
    """Save an RGB video with pixel-space tracks overlaid."""
    frames_rgb = np.asarray(frames_rgb)
    tracks = np.asarray(tracks, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T, H, W, 3), got {frames_rgb.shape}")
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Expected tracks with shape (T, N, 2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"Track/visibility mismatch: tracks={tracks.shape}, visibility={visibility.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(frames_rgb.shape[0], tracks.shape[0], visibility.shape[0])
    height, width = frames_rgb.shape[1:3]
    if not np.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"Overlay scale must be positive, got {scale}")
    scale = float(scale)
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    out_width += out_width % 2
    out_height += out_height % 2
    draw_tracks = tracks.copy()
    draw_tracks[..., 0] *= out_width / float(width)
    draw_tracks[..., 1] *= out_height / float(height)
    radius = max(1, int(round(radius * scale)))
    line_width = max(1, int(round(line_width * scale)))
    write_path = output_path
    temp_path: Path | None = None
    if crf is not None and shutil.which("ffmpeg") is not None:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            prefix=f"{output_path.stem}_raw_",
            dir=str(output_path.parent),
            delete=False,
        )
        temp_file.close()
        temp_path = Path(temp_file.name)
        write_path = temp_path
    writer = cv2.VideoWriter(
        str(write_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(out_width), int(out_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {write_path}")

    try:
        for frame_idx in range(frame_count):
            canvas = cv2.resize(
                frames_rgb[frame_idx],
                (out_width, out_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            canvas = np.ascontiguousarray(canvas)
            start_idx = 0 if trace_frames < 0 else max(0, frame_idx - int(trace_frames) + 1)
            for point_idx in range(tracks.shape[1]):
                color = _point_color(point_idx)
                history: list[tuple[int, int]] = []
                for hist_idx in range(start_idx, frame_idx + 1):
                    if visibility[hist_idx, point_idx] <= 0.5:
                        continue
                    x, y = draw_tracks[hist_idx, point_idx]
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    if 0 <= xi < out_width and 0 <= yi < out_height:
                        history.append((xi, yi))
                if len(history) >= 2:
                    pts = np.asarray(history, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=line_width)
                if history:
                    cv2.circle(canvas, history[-1], radius, color, thickness=-1)

            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if temp_path is not None:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(temp_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(int(crf)),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
        finally:
            temp_path.unlink(missing_ok=True)
