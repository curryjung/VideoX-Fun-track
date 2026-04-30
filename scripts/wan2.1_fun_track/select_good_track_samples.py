#!/usr/bin/env python
"""Interactive selector for latent-track metadata samples.

This tool lets you review latent-only samples visually using:
- first_frame.png
- transformed track npz (drawn as moving points/trajectories)

Then it saves selected rows back to metadata JSON format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np


DEFAULT_TRACK_KEYS: Tuple[str, ...] = ("tracks_compressed", "tracks")
DEFAULT_VIS_KEYS: Tuple[str, ...] = ("visibility_compressed", "visibility")


@dataclass
class SampleInfo:
    meta_index: int
    row: Dict[str, Any]
    latent_rel: str
    latent_abs: str
    track_rel: str
    track_abs: str
    first_frame_abs: str
    sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review latent track samples and save selected metadata rows."
    )
    parser.add_argument(
        "--meta_json",
        type=str,
        required=True,
        help="Input metadata JSON (list of rows).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root for resolving relative paths.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="datasets/internal_datasets/metadata_track_fineMotion_train_selected64.json",
        help="Output JSON path for selected rows.",
    )
    parser.add_argument(
        "--state_json",
        type=str,
        default="datasets/internal_datasets/metadata_track_fineMotion_train_selected64.state.json",
        help="State file path for resume.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/videox_fun_track_selector_cache",
        help="Directory for generated preview videos.",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=64,
        help="Target number of selected samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle review order.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=81,
        help="Maximum frames in rendered preview video.",
    )
    parser.add_argument(
        "--preview_fps",
        type=int,
        default=8,
        help="FPS for rendered preview video.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200,
        help="Max number of track points drawn per sample.",
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=8,
        help="Trajectory length in frames.",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=2,
        help="Point radius for track overlay.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for Gradio UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port for Gradio UI.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link.",
    )
    return parser.parse_args()


def _resolve_path(path_value: str, data_root: str) -> str:
    if path_value == "":
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(data_root, path_value)


def _infer_sample_id(latent_rel: str) -> str:
    path = Path(latent_rel)
    parts = path.parts
    if len(parts) >= 3 and parts[-2].startswith("processed_"):
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return path.stem


def load_samples(meta_json: str, data_root: str) -> List[SampleInfo]:
    with open(meta_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected metadata list: {meta_json}")

    samples: List[SampleInfo] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        latent_rel = str(row.get("latent_file_path", row.get("file_path", "")))
        track_rel = str(row.get("track_file_path", ""))
        if latent_rel == "" or track_rel == "":
            continue

        latent_abs = _resolve_path(latent_rel, data_root)
        track_abs = _resolve_path(track_rel, data_root)

        first_frame_rel = str(row.get("first_frame_file_path", ""))
        if first_frame_rel != "":
            first_frame_abs = _resolve_path(first_frame_rel, data_root)
        else:
            first_frame_abs = os.path.join(os.path.dirname(latent_abs), "first_frame.png")

        samples.append(
            SampleInfo(
                meta_index=i,
                row=row,
                latent_rel=latent_rel,
                latent_abs=latent_abs,
                track_rel=track_rel,
                track_abs=track_abs,
                first_frame_abs=first_frame_abs,
                sample_id=_infer_sample_id(latent_rel),
            )
        )
    return samples


def _load_track_arrays(track_abs: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(track_abs)
    tracks = None
    for key in DEFAULT_TRACK_KEYS:
        if key in data:
            tracks = data[key]
            break
    if tracks is None:
        raise KeyError(f"Track key missing in {track_abs}")

    visibility = None
    for key in DEFAULT_VIS_KEYS:
        if key in data:
            visibility = data[key]
            break

    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_abs}")

    if visibility is None:
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim != 2:
        raise ValueError(f"Unexpected visibility shape {visibility.shape} in {track_abs}")

    tracks = tracks.astype(np.float32, copy=False)
    visibility = visibility.astype(np.float32, copy=False)
    return tracks, visibility


def _pick_point_indices(visibility: np.ndarray, max_points: int) -> np.ndarray:
    n_points = visibility.shape[1]
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    vis_score = np.mean(visibility > 0.5, axis=0)
    order = np.argsort(-vis_score)
    return order[:max_points].astype(np.int64)


def _compute_metrics(tracks: np.ndarray, visibility: np.ndarray) -> Dict[str, float]:
    vis_binary = visibility > 0.5
    visibility_ratio = float(np.mean(vis_binary))
    occlusion_ratio = 1.0 - visibility_ratio

    delta = tracks[1:] - tracks[:-1]
    speed = np.linalg.norm(delta, axis=-1)
    valid_speed = vis_binary[1:] & vis_binary[:-1]
    speed_values = speed[valid_speed]
    if speed_values.size == 0:
        mean_speed = 0.0
        p95_speed = 0.0
    else:
        mean_speed = float(np.mean(speed_values))
        p95_speed = float(np.percentile(speed_values, 95.0))

    accel = delta[1:] - delta[:-1]
    jitter = np.linalg.norm(accel, axis=-1)
    valid_jitter = vis_binary[2:] & vis_binary[1:-1] & vis_binary[:-2]
    jitter_values = jitter[valid_jitter]
    jitter_mean = float(np.mean(jitter_values)) if jitter_values.size > 0 else 0.0

    return {
        "visibility_ratio": visibility_ratio,
        "occlusion_ratio": occlusion_ratio,
        "mean_speed": mean_speed,
        "p95_speed": p95_speed,
        "jitter_mean": jitter_mean,
    }


def _make_color_table(n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    hues = np.linspace(0, 179, num=n_points, endpoint=False, dtype=np.float32)
    hsv = np.stack(
        [hues, np.full_like(hues, 180.0), np.full_like(hues, 255.0)],
        axis=1,
    ).reshape(1, n_points, 3).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n_points, 3)
    return bgr


def _draw_one_frame(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    point_ids: np.ndarray,
    colors: np.ndarray,
    t: int,
    trail_length: int,
    point_radius: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    t0 = max(0, t - max(trail_length, 1) + 1)
    for local_idx, point_id in enumerate(point_ids):
        color = (
            int(colors[local_idx, 0]),
            int(colors[local_idx, 1]),
            int(colors[local_idx, 2]),
        )

        for k in range(t0 + 1, t + 1):
            if visibility[k - 1, point_id] <= 0.5 or visibility[k, point_id] <= 0.5:
                continue
            x0, y0 = tracks[k - 1, point_id]
            x1, y1 = tracks[k, point_id]
            if not np.isfinite([x0, y0, x1, y1]).all():
                continue
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            cv2.line(
                out,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                color,
                1,
                cv2.LINE_AA,
            )

        if visibility[t, point_id] <= 0.5:
            continue
        x, y = tracks[t, point_id]
        if not np.isfinite([x, y]).all():
            continue
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                out,
                (int(round(x)), int(round(y))),
                max(1, point_radius),
                color,
                -1,
                cv2.LINE_AA,
            )
    return out


class TrackSelectorContext:
    def __init__(self, args: argparse.Namespace, samples: List[SampleInfo]):
        self.args = args
        self.samples = samples
        self.cache_dir = Path(args.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preview_cache: Dict[int, str] = {}
        self.metrics_cache: Dict[int, Dict[str, float]] = {}
        self.error_cache: Dict[int, str] = {}

    def _preview_key(self, sample: SampleInfo) -> str:
        signature = (
            f"{sample.track_abs}|{sample.first_frame_abs}|"
            f"{self.args.max_frames}|{self.args.preview_fps}|{self.args.max_points}|"
            f"{self.args.trail_length}|{self.args.point_radius}"
        )
        return hashlib.md5(signature.encode("utf-8")).hexdigest()

    def _render_preview(self, sample: SampleInfo, out_path: Path) -> Dict[str, float]:
        tracks, visibility = _load_track_arrays(sample.track_abs)
        metrics = _compute_metrics(tracks, visibility)

        n_frames = tracks.shape[0]
        if self.args.max_frames > 0:
            n_frames = min(n_frames, self.args.max_frames)
        tracks = tracks[:n_frames]
        visibility = visibility[:n_frames]

        base = cv2.imread(sample.first_frame_abs, cv2.IMREAD_COLOR)
        if base is None:
            x_max = int(np.nanmax(tracks[..., 0])) if np.isfinite(tracks[..., 0]).any() else 832
            y_max = int(np.nanmax(tracks[..., 1])) if np.isfinite(tracks[..., 1]).any() else 480
            width = int(np.clip(x_max + 1, 64, 1920))
            height = int(np.clip(y_max + 1, 64, 1080))
            base = np.zeros((height, width, 3), dtype=np.uint8)

        height, width = base.shape[:2]
        if width % 2 == 1:
            width -= 1
        if height % 2 == 1:
            height -= 1
        base = base[:height, :width]

        point_ids = _pick_point_indices(visibility, self.args.max_points)
        colors = _make_color_table(len(point_ids))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            float(max(1, self.args.preview_fps)),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {out_path}")

        try:
            for t in range(n_frames):
                frame = _draw_one_frame(
                    frame=base,
                    tracks=tracks,
                    visibility=visibility,
                    point_ids=point_ids,
                    colors=colors,
                    t=t,
                    trail_length=self.args.trail_length,
                    point_radius=self.args.point_radius,
                )
                cv2.putText(
                    frame,
                    f"t={t + 1}/{n_frames} pts={len(point_ids)}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()

        return metrics

    def ensure_preview(self, meta_index: int) -> Tuple[Optional[str], Dict[str, float], str]:
        if meta_index in self.preview_cache:
            preview = self.preview_cache[meta_index]
            metrics = self.metrics_cache.get(meta_index, {})
            return preview, metrics, self.error_cache.get(meta_index, "")

        sample = self.samples[meta_index]
        out_path = self.cache_dir / f"{self._preview_key(sample)}.mp4"
        try:
            if not out_path.exists():
                metrics = self._render_preview(sample, out_path)
            else:
                tracks, visibility = _load_track_arrays(sample.track_abs)
                metrics = _compute_metrics(tracks, visibility)
            self.preview_cache[meta_index] = str(out_path)
            self.metrics_cache[meta_index] = metrics
            self.error_cache[meta_index] = ""
            return str(out_path), metrics, ""
        except Exception as exc:
            self.preview_cache[meta_index] = None  # type: ignore[assignment]
            self.metrics_cache[meta_index] = {}
            self.error_cache[meta_index] = str(exc)
            return None, {}, str(exc)


def _init_state(num_samples: int, seed: int, shuffle: bool) -> Dict[str, Any]:
    order = list(range(num_samples))
    if shuffle:
        random.Random(seed).shuffle(order)
    return {
        "order": order,
        "cursor": 0,
        "kept": [],
        "rejected": [],
    }


def _state_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_state(path: str, num_samples: int, seed: int, shuffle: bool) -> Dict[str, Any]:
    state_file = _state_path(path)
    if not state_file.exists():
        return _init_state(num_samples, seed, shuffle)
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        return _init_state(num_samples, seed, shuffle)
    if not {"order", "cursor", "kept", "rejected"} <= set(state.keys()):
        return _init_state(num_samples, seed, shuffle)

    order = [int(i) for i in state.get("order", []) if 0 <= int(i) < num_samples]
    if len(order) != num_samples:
        order = list(range(num_samples))
        if shuffle:
            random.Random(seed).shuffle(order)
    kept = sorted(set(int(i) for i in state.get("kept", []) if 0 <= int(i) < num_samples))
    rejected = sorted(set(int(i) for i in state.get("rejected", []) if 0 <= int(i) < num_samples))
    cursor = int(state.get("cursor", 0))
    cursor = max(0, min(cursor, max(0, len(order) - 1)))
    return {
        "order": order,
        "cursor": cursor,
        "kept": kept,
        "rejected": rejected,
    }


def save_state(path: str, state: Dict[str, Any]) -> None:
    state_file = _state_path(path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _decision_of(meta_index: int, state: Dict[str, Any]) -> str:
    kept = set(state["kept"])
    rejected = set(state["rejected"])
    if meta_index in kept:
        return "KEEP"
    if meta_index in rejected:
        return "REJECT"
    return "UNDECIDED"


def _safe_caption(text: str, limit: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _build_selected_table(state: Dict[str, Any], samples: Sequence[SampleInfo]) -> List[List[Any]]:
    kept = set(state["kept"])
    order = state["order"]
    order_pos = {meta_idx: pos for pos, meta_idx in enumerate(order)}
    rows: List[List[Any]] = []
    for rank, meta_idx in enumerate(sorted(kept, key=lambda x: order_pos.get(x, x)), start=1):
        sample = samples[meta_idx]
        rows.append(
            [
                rank,
                meta_idx,
                sample.sample_id,
                sample.track_rel,
            ]
        )
    return rows


def _format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "-"
    return (
        f"vis={metrics.get('visibility_ratio', 0.0):.3f} | "
        f"occ={metrics.get('occlusion_ratio', 0.0):.3f} | "
        f"speed_mean={metrics.get('mean_speed', 0.0):.2f} | "
        f"speed_p95={metrics.get('p95_speed', 0.0):.2f} | "
        f"jitter={metrics.get('jitter_mean', 0.0):.2f}"
    )


def _load_first_frame_rgb(image_path: str) -> Optional[np.ndarray]:
    if not os.path.isfile(image_path):
        return None
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _move_cursor(state: Dict[str, Any], delta: int) -> Dict[str, Any]:
    n = len(state["order"])
    cursor = int(state["cursor"]) + int(delta)
    cursor = max(0, min(cursor, max(0, n - 1)))
    state["cursor"] = cursor
    return state


def _goto_cursor(state: Dict[str, Any], one_based_position: float) -> Dict[str, Any]:
    n = len(state["order"])
    pos = int(one_based_position) - 1
    pos = max(0, min(pos, max(0, n - 1)))
    state["cursor"] = pos
    return state


def _apply_decision(
    state: Dict[str, Any],
    decision: str,
    target_count: int,
    auto_next: bool = True,
) -> Tuple[Dict[str, Any], str]:
    if len(state["order"]) == 0:
        return state, "No samples."
    cursor = int(state["cursor"])
    meta_idx = int(state["order"][cursor])
    kept = set(int(i) for i in state["kept"])
    rejected = set(int(i) for i in state["rejected"])

    if decision == "keep":
        if target_count > 0 and len(kept) >= target_count and meta_idx not in kept:
            return state, f"Target {target_count} already reached."
        kept.add(meta_idx)
        rejected.discard(meta_idx)
    elif decision == "reject":
        rejected.add(meta_idx)
        kept.discard(meta_idx)

    state["kept"] = sorted(kept)
    state["rejected"] = sorted(rejected)
    if auto_next:
        state = _move_cursor(state, +1)
    return state, ""


def save_selected_json(
    output_json: str,
    state: Dict[str, Any],
    samples: Sequence[SampleInfo],
    target_count: int,
) -> Tuple[str, int]:
    kept = set(int(i) for i in state["kept"])
    order = state["order"]
    order_pos = {meta_idx: pos for pos, meta_idx in enumerate(order)}
    kept_sorted = sorted(kept, key=lambda x: order_pos.get(x, x))
    if target_count > 0:
        kept_sorted = kept_sorted[:target_count]
    out_rows = [samples[idx].row for idx in kept_sorted]

    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    return str(output_path), len(out_rows)


def main() -> None:
    args = parse_args()
    meta_json = str(Path(args.meta_json).expanduser().resolve())
    data_root = str(Path(args.data_root).expanduser().resolve())

    samples = load_samples(meta_json=meta_json, data_root=data_root)
    if len(samples) == 0:
        raise RuntimeError("No valid samples found in metadata.")

    context = TrackSelectorContext(args=args, samples=samples)
    initial_state = load_state(
        path=args.state_json,
        num_samples=len(samples),
        seed=args.seed,
        shuffle=args.shuffle,
    )

    def render(state: Dict[str, Any], note: str = ""):
        if len(state["order"]) == 0:
            return (
                None,
                None,
                "No samples.",
                "No samples.",
                [],
                state,
            )

        cursor = int(state["cursor"])
        meta_idx = int(state["order"][cursor])
        sample = samples[meta_idx]
        preview_path, metrics, err = context.ensure_preview(meta_idx)
        decision = _decision_of(meta_idx, state)
        selected_count = len(state["kept"])

        first_frame_value = _load_first_frame_rgb(sample.first_frame_abs)
        caption = _safe_caption(str(sample.row.get("text", "")))

        info_lines = [
            f"**Order**: {cursor + 1}/{len(state['order'])}",
            f"**Meta index**: {meta_idx}",
            f"**Decision**: `{decision}`",
            f"**Sample ID**: `{sample.sample_id}`",
            f"**Track**: `{sample.track_rel}`",
            f"**Metrics**: `{_format_metrics(metrics)}`",
            f"**Caption**: {caption if caption else '-'}",
        ]
        if err:
            info_lines.append(f"**Preview error**: `{err}`")
        if note:
            info_lines.append(f"**Note**: {note}")

        counter_text = (
            f"Selected: **{selected_count}** / target **{args.target_count}**  "
            f"(Rejected: {len(state['rejected'])})"
        )
        table_rows = _build_selected_table(state=state, samples=samples)

        return (
            preview_path,
            first_frame_value,
            "\n".join(info_lines),
            counter_text,
            table_rows,
            state,
        )

    def on_prev(state: Dict[str, Any]):
        state = _move_cursor(state, -1)
        save_state(args.state_json, state)
        return render(state)

    def on_next(state: Dict[str, Any]):
        state = _move_cursor(state, +1)
        save_state(args.state_json, state)
        return render(state)

    def on_keep(state: Dict[str, Any]):
        state, note = _apply_decision(state, decision="keep", target_count=args.target_count, auto_next=True)
        save_state(args.state_json, state)
        return render(state, note=note)

    def on_reject(state: Dict[str, Any]):
        state, note = _apply_decision(state, decision="reject", target_count=args.target_count, auto_next=True)
        save_state(args.state_json, state)
        return render(state, note=note)

    def on_go(position: float, state: Dict[str, Any]):
        state = _goto_cursor(state, position)
        save_state(args.state_json, state)
        return render(state)

    def on_save_selected(state: Dict[str, Any]):
        out_path, count = save_selected_json(
            output_json=args.output_json,
            state=state,
            samples=samples,
            target_count=args.target_count,
        )
        save_state(args.state_json, state)
        return render(state, note=f"Saved {count} rows -> {out_path}")

    with gr.Blocks(title="VideoX-Fun Track Sample Selector") as demo:
        gr.Markdown(
            "# VideoX-Fun Track Sample Selector\n"
            "latent-only 데이터에서 `first_frame + track` 미리보기를 보며 샘플을 고릅니다."
        )

        state_store = gr.State(initial_state)

        with gr.Row():
            prev_button = gr.Button("Prev")
            keep_button = gr.Button("Keep")
            reject_button = gr.Button("Reject")
            next_button = gr.Button("Next")
            save_button = gr.Button("Save Selected JSON", variant="primary")

        with gr.Row():
            go_position_input = gr.Number(
                label="Go to order position (1-based)",
                value=1,
                precision=0,
            )
            go_button = gr.Button("Go")

        with gr.Row():
            video_preview = gr.Video(label="Track Overlay Preview", height=420, autoplay=False)
            first_frame_preview = gr.Image(label="First Frame", height=420)

        info_markdown = gr.Markdown()
        counter_markdown = gr.Markdown()
        selected_table = gr.Dataframe(
            headers=["rank", "meta_index", "sample_id", "track_file_path"],
            datatype=["number", "number", "str", "str"],
            row_count=(0, "dynamic"),
            column_count=(4, "fixed"),
            interactive=False,
            wrap=True,
            label="Selected Samples",
        )

        outputs = [
            video_preview,
            first_frame_preview,
            info_markdown,
            counter_markdown,
            selected_table,
            state_store,
        ]

        demo.load(
            fn=render,
            inputs=[state_store],
            outputs=outputs,
        )
        prev_button.click(fn=on_prev, inputs=[state_store], outputs=outputs)
        next_button.click(fn=on_next, inputs=[state_store], outputs=outputs)
        keep_button.click(fn=on_keep, inputs=[state_store], outputs=outputs)
        reject_button.click(fn=on_reject, inputs=[state_store], outputs=outputs)
        go_button.click(fn=on_go, inputs=[go_position_input, state_store], outputs=outputs)
        save_button.click(fn=on_save_selected, inputs=[state_store], outputs=outputs)

    print(f"[info] meta_json={meta_json}")
    print(f"[info] data_root={data_root}")
    print(f"[info] samples={len(samples)}")
    print(f"[info] state_json={Path(args.state_json).expanduser().resolve()}")
    print(f"[info] output_json={Path(args.output_json).expanduser().resolve()}")
    print(f"[info] cache_dir={context.cache_dir}")
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
