#!/usr/bin/env python
"""Web viewer for browsing experiment videos under samples directory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import gradio as gr


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


@dataclass
class VideoItem:
    """Metadata for one video file in the samples tree."""

    relative_path: str
    absolute_path: str
    size_mb: float
    modified_at: str
    modified_ts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a lightweight web UI for sample video browsing.",
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "samples"),
        help="Root directory to scan recursively for videos.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for Gradio server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing link.",
    )
    return parser.parse_args()


def _iter_video_paths(samples_dir: Path) -> Iterable[Path]:
    for file_path in samples_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in VIDEO_EXTENSIONS:
            yield file_path


def scan_videos(samples_dir: str, keyword: str, max_results: int) -> list[VideoItem]:
    root_path = Path(samples_dir).expanduser().resolve()
    if not root_path.exists():
        raise gr.Error(f"Directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise gr.Error(f"Not a directory: {root_path}")

    normalized_keyword = keyword.strip().lower()
    video_items: list[VideoItem] = []
    for file_path in _iter_video_paths(root_path):
        relative_path = file_path.relative_to(root_path).as_posix()
        if normalized_keyword and normalized_keyword not in relative_path.lower():
            continue
        stat = file_path.stat()
        video_items.append(
            VideoItem(
                relative_path=relative_path,
                absolute_path=str(file_path),
                size_mb=stat.st_size / (1024 * 1024),
                modified_at=datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                modified_ts=stat.st_mtime,
            )
        )

    video_items.sort(key=lambda item: item.modified_ts, reverse=True)
    if max_results > 0:
        video_items = video_items[:max_results]
    return video_items


def build_table_rows(video_items: list[VideoItem]) -> list[list[str | int]]:
    rows: list[list[str | int]] = []
    for row_index, item in enumerate(video_items):
        rows.append(
            [
                row_index,
                item.relative_path,
                f"{item.size_mb:.2f}",
                item.modified_at,
            ]
        )
    return rows


def refresh_video_list(
    samples_dir: str,
    keyword: str,
    max_results: int,
) -> tuple[list[list[str | int]], list[str], list[str], str | None, str, str]:
    video_items = scan_videos(
        samples_dir=samples_dir,
        keyword=keyword,
        max_results=max_results,
    )
    table_rows = build_table_rows(video_items)
    absolute_paths = [item.absolute_path for item in video_items]
    relative_paths = [item.relative_path for item in video_items]

    if not video_items:
        status_text = (
            "검색 결과가 없습니다. "
            "keyword를 비우거나 samples_dir 경로를 확인해 주세요."
        )
        return table_rows, absolute_paths, relative_paths, None, "", status_text

    first_item = video_items[0]
    status_text = f"총 {len(video_items)}개 비디오를 찾았습니다."
    return (
        table_rows,
        absolute_paths,
        relative_paths,
        first_item.absolute_path,
        first_item.relative_path,
        status_text,
    )


def select_video_from_table(
    select_data: Any,
    absolute_paths: list[str],
    relative_paths: list[str],
) -> tuple[str | None, str, str]:
    if not absolute_paths:
        return None, "", "선택 가능한 비디오가 없습니다."

    selected_index = select_data.index
    row_index = (
        selected_index[0]
        if isinstance(selected_index, tuple)
        else int(selected_index)
    )
    if row_index < 0 or row_index >= len(absolute_paths):
        return None, "", "유효하지 않은 선택입니다."

    selected_absolute_path = absolute_paths[row_index]
    selected_relative_path = relative_paths[row_index]
    status_text = f"{row_index + 1}/{len(absolute_paths)} 번째 비디오 선택됨"
    return selected_absolute_path, selected_relative_path, status_text


def build_ui(initial_samples_dir: str) -> gr.Blocks:
    with gr.Blocks(title="VideoX-Fun Samples Video Viewer") as demo:
        gr.Markdown(
            "# VideoX-Fun Samples Video Viewer\n"
            "samples 폴더를 재귀적으로 스캔해 실험 비디오를 빠르게 확인합니다."
        )

        absolute_paths_state = gr.State([])
        relative_paths_state = gr.State([])

        with gr.Row():
            samples_dir_input = gr.Textbox(
                label="samples_dir",
                value=initial_samples_dir,
                lines=1,
            )
            keyword_input = gr.Textbox(
                label="keyword filter (relative path contains)",
                value="",
                lines=1,
            )
            max_results_input = gr.Slider(
                label="max results",
                minimum=50,
                maximum=5000,
                step=50,
                value=2000,
            )
            refresh_button = gr.Button("Refresh", variant="primary")

        with gr.Row():
            video_table = gr.Dataframe(
                headers=["idx", "relative_path", "size_mb", "modified_at"],
                datatype=["number", "str", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(4, "fixed"),
                interactive=False,
                wrap=True,
                label="video list (click row to preview)",
            )
            video_preview = gr.Video(
                label="selected video preview",
                autoplay=False,
                height=420,
            )

        selected_path_output = gr.Textbox(
            label="selected relative path",
            lines=1,
            interactive=False,
        )
        status_output = gr.Markdown("Ready")

        refresh_outputs = [
            video_table,
            absolute_paths_state,
            relative_paths_state,
            video_preview,
            selected_path_output,
            status_output,
        ]
        refresh_button.click(
            fn=refresh_video_list,
            inputs=[samples_dir_input, keyword_input, max_results_input],
            outputs=refresh_outputs,
        )
        keyword_input.submit(
            fn=refresh_video_list,
            inputs=[samples_dir_input, keyword_input, max_results_input],
            outputs=refresh_outputs,
        )
        video_table.select(
            fn=select_video_from_table,
            inputs=[absolute_paths_state, relative_paths_state],
            outputs=[video_preview, selected_path_output, status_output],
        )

        demo.load(
            fn=refresh_video_list,
            inputs=[samples_dir_input, keyword_input, max_results_input],
            outputs=refresh_outputs,
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = build_ui(initial_samples_dir=args.samples_dir)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
