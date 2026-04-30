#!/usr/bin/env python
"""Serve a sample-wise comparison webpage for track analysis videos.

Rows are samples and columns are:
1) joint
2) motion only
3) text only
"""

from __future__ import annotations

import argparse
import html
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, urlparse


MODE_COLUMNS: list[tuple[str, str]] = [
    ("joint", "analysis_from_val_metadata_joint"),
    ("motion_only", "analysis_from_val_metadata_motion_only"),
    ("text_only", "analysis_from_val_metadata_text_only"),
]
MODE_TITLES = {
    "joint": "joint",
    "motion_only": "motion only",
    "text_only": "text only",
}
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
TOKEN_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve comparison page for joint/motion/text track videos.",
    )
    parser.add_argument(
        "--experiments_root",
        type=str,
        default="",
        help=(
            "Root directory that contains multiple experiment directories. "
            "If empty, parent of --experiment_dir is used."
        ),
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="",
        help=(
            "Default selected experiment directory. "
            "If empty, first valid experiment under root is selected."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint folder name (e.g. checkpoint-200). Empty selects latest.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["overlay", "raw"],
        default="overlay",
        help="Video variant preference: overlay uses *_overlay.mp4 when possible.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8780,
        help="Port for the web server.",
    )
    return parser.parse_args()


def natural_key(text: str) -> list[object]:
    tokens = TOKEN_PATTERN.split(text)
    key: list[object] = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token.lower())
    return key


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    number = int(match.group(1)) if match else -1
    return number, path.stat().st_mtime, path.name


def list_checkpoints(experiment_dir: Path) -> list[Path]:
    checkpoints = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name)
    ]
    checkpoints.sort(key=checkpoint_sort_key)
    return checkpoints


def has_checkpoints(experiment_dir: Path) -> bool:
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        return False
    return any(
        child.is_dir() and CHECKPOINT_PATTERN.fullmatch(child.name)
        for child in experiment_dir.iterdir()
    )


def list_experiments(experiments_root: Path) -> list[Path]:
    experiments: list[Path] = []
    if has_checkpoints(experiments_root):
        experiments.append(experiments_root)

    for path in experiments_root.iterdir():
        if path.is_dir() and has_checkpoints(path):
            experiments.append(path)

    experiments.sort(
        key=lambda path: natural_key(path.relative_to(experiments_root).as_posix())
    )
    return experiments


def experiment_query_value(experiment_dir: Path, experiments_root: Path) -> str:
    return experiment_dir.relative_to(experiments_root).as_posix()


def select_experiment(
    experiments_root: Path,
    experiments: list[Path],
    requested_experiment: str,
    default_experiment: str,
) -> Path:
    if not experiments:
        raise ValueError(
            f"No experiment directory found under root: {experiments_root}"
        )

    candidates = [requested_experiment.strip(), default_experiment.strip()]
    for candidate in candidates:
        if not candidate:
            continue
        for experiment_dir in experiments:
            experiment_rel = experiment_query_value(experiment_dir, experiments_root)
            if candidate in {experiment_rel, experiment_dir.name, str(experiment_dir)}:
                return experiment_dir

    return experiments[0]


def select_checkpoint(
    checkpoints: list[Path],
    requested_checkpoint: str,
    default_checkpoint: str,
) -> Path:
    if not checkpoints:
        raise ValueError("No checkpoint-* directories found in selected experiment.")

    candidates = [requested_checkpoint.strip(), default_checkpoint.strip()]
    for candidate in candidates:
        if not candidate:
            continue
        for checkpoint_path in checkpoints:
            if checkpoint_path.name == candidate:
                return checkpoint_path

    return checkpoints[-1]


def iter_sample_dirs(mode_dir: Path) -> Iterable[Path]:
    if not mode_dir.exists():
        return []
    return sorted((path for path in mode_dir.iterdir() if path.is_dir()), key=lambda p: natural_key(p.name))


def pick_video(sample_dir: Path, prefer_overlay: bool) -> Path | None:
    all_mp4 = [path for path in sample_dir.rglob("*.mp4") if path.is_file()]
    if not all_mp4:
        return None

    overlay_files = [path for path in all_mp4 if path.name.endswith("_overlay.mp4")]
    raw_files = [path for path in all_mp4 if not path.name.endswith("_overlay.mp4")]

    preferred = overlay_files if prefer_overlay else raw_files
    fallback = raw_files if prefer_overlay else overlay_files
    candidates = preferred if preferred else fallback
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_rows(
    checkpoint_dir: Path,
    experiments_root: Path,
    prefer_overlay: bool,
) -> tuple[list[str], list[dict[str, str | None]], dict[str, int]]:
    mode_to_sample_video: dict[str, dict[str, str]] = {mode: {} for mode, _ in MODE_COLUMNS}
    sample_names: set[str] = set()

    for mode_name, mode_subdir in MODE_COLUMNS:
        mode_dir = checkpoint_dir / mode_subdir
        for sample_dir in iter_sample_dirs(mode_dir):
            selected_video = pick_video(sample_dir=sample_dir, prefer_overlay=prefer_overlay)
            if selected_video is None:
                continue
            sample_name = sample_dir.name
            sample_names.add(sample_name)
            relative_video = selected_video.relative_to(experiments_root).as_posix()
            mode_to_sample_video[mode_name][sample_name] = relative_video

    sorted_samples = sorted(sample_names, key=natural_key)
    rows: list[dict[str, str | None]] = []
    for sample_name in sorted_samples:
        row: dict[str, str | None] = {"sample": sample_name}
        for mode_name, _ in MODE_COLUMNS:
            row[mode_name] = mode_to_sample_video[mode_name].get(sample_name)
        rows.append(row)

    mode_counts = {
        mode_name: sum(1 for row in rows if row.get(mode_name))
        for mode_name, _ in MODE_COLUMNS
    }
    return sorted_samples, rows, mode_counts


def video_cell(video_rel_path: str | None) -> str:
    if video_rel_path is None:
        return "<div class='missing'>N/A</div>"
    escaped_label = html.escape(Path(video_rel_path).name)
    escaped_src = "/" + quote(video_rel_path, safe="/")
    return (
        "<div class='video-cell'>"
        f"<video controls preload='metadata' playsinline muted src='{escaped_src}'></video>"
        f"<div class='filename'>{escaped_label}</div>"
        "</div>"
    )


def render_experiment_options(
    experiments_root: Path,
    experiments: list[Path],
    selected_experiment: Path,
) -> str:
    selected_value = experiment_query_value(selected_experiment, experiments_root)
    option_tags: list[str] = []
    for experiment_dir in experiments:
        option_value = experiment_query_value(experiment_dir, experiments_root)
        is_selected = " selected" if option_value == selected_value else ""
        label = (
            option_value
            if option_value != "."
            else experiment_dir.name
        )
        option_tags.append(
            f"<option value='{html.escape(option_value)}'{is_selected}>"
            f"{html.escape(label)}"
            "</option>"
        )
    return "\n".join(option_tags)


def render_checkpoint_options(
    checkpoints: list[Path],
    selected_checkpoint: Path,
) -> str:
    option_tags: list[str] = []
    for checkpoint_dir in checkpoints:
        is_selected = " selected" if checkpoint_dir.name == selected_checkpoint.name else ""
        option_tags.append(
            f"<option value='{html.escape(checkpoint_dir.name)}'{is_selected}>"
            f"{html.escape(checkpoint_dir.name)}"
            "</option>"
        )
    return "\n".join(option_tags)


def render_page(
    experiments_root: Path,
    experiments: list[Path],
    selected_experiment: Path,
    selected_checkpoint: Path,
    checkpoints: list[Path],
    variant: str,
) -> str:
    prefer_overlay = variant == "overlay"
    _, rows, mode_counts = build_rows(
        checkpoint_dir=selected_checkpoint,
        experiments_root=experiments_root,
        prefer_overlay=prefer_overlay,
    )

    body_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        body_rows.append(
            "<tr>"
            f"<td class='sample-col'>{row_index}. {html.escape(str(row['sample']))}</td>"
            f"<td>{video_cell(row.get('joint'))}</td>"
            f"<td>{video_cell(row.get('motion_only'))}</td>"
            f"<td>{video_cell(row.get('text_only'))}</td>"
            "</tr>"
        )

    table_html = (
        "\n".join(body_rows)
        if body_rows
        else "<tr><td colspan='4' class='missing'>No rows found.</td></tr>"
    )
    experiment_options = render_experiment_options(
        experiments_root=experiments_root,
        experiments=experiments,
        selected_experiment=selected_experiment,
    )
    checkpoint_options = render_checkpoint_options(
        checkpoints=checkpoints,
        selected_checkpoint=selected_checkpoint,
    )
    variant_overlay_selected = " selected" if variant == "overlay" else ""
    variant_raw_selected = " selected" if variant == "raw" else ""
    selected_experiment_query = experiment_query_value(
        selected_experiment,
        experiments_root,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Track Analysis Comparison Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111317;
      --card: #1a1e24;
      --text: #e8ebf0;
      --muted: #9aa4b2;
      --border: #2d3440;
      --accent: #5aa8ff;
    }}
    body {{
      margin: 0;
      padding: 20px;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, system-ui, -apple-system, sans-serif;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 24px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 12px;
      font-size: 14px;
      line-height: 1.4;
      word-break: break-all;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      align-items: center;
    }}
    .label {{
      color: var(--muted);
      margin-right: 4px;
      font-size: 13px;
    }}
    .control-form {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      background: var(--card);
      padding: 10px;
      border-radius: 8px;
    }}
    .control-form label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .control-form select {{
      min-width: 220px;
      max-width: 520px;
      background: #10141a;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 13px;
    }}
    .control-form button {{
      background: #1f6feb;
      color: white;
      border: none;
      border-radius: 7px;
      padding: 7px 12px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
    }}
    .help-text {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 12px;
    }}
    .summary {{
      color: var(--muted);
      margin: 10px 0 14px 0;
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--card);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1400px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #1f2530;
      text-transform: none;
      letter-spacing: 0.2px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    .sample-col {{
      width: 220px;
      font-weight: 600;
      color: #cdd7e3;
      white-space: nowrap;
    }}
    .video-cell {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    video {{
      width: 100%;
      max-width: 420px;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: black;
    }}
    .filename {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .missing {{
      color: #9aa4b2;
      font-size: 13px;
      padding: 20px 8px;
    }}
  </style>
</head>
<body>
  <h1>Track Analysis Comparison Viewer</h1>
  <div class="meta">
    experiments_root: {html.escape(str(experiments_root))}<br />
    experiment: {html.escape(str(selected_experiment))}<br />
    checkpoint: {html.escape(selected_checkpoint.name)}
  </div>

  <div class="toolbar">
    <form class="control-form" method="get">
      <input type="hidden" name="source_experiment" value="{html.escape(selected_experiment_query)}" />
      <label for="experiment">experiment_dir</label>
      <select id="experiment" name="experiment">
        {experiment_options}
      </select>

      <label for="checkpoint">checkpoint</label>
      <select id="checkpoint" name="checkpoint">
        {checkpoint_options}
      </select>

      <label for="variant">variant</label>
      <select id="variant" name="variant">
        <option value="overlay"{variant_overlay_selected}>overlay</option>
        <option value="raw"{variant_raw_selected}>raw</option>
      </select>

      <button type="submit">Apply</button>
    </form>
  </div>
  <div class="help-text">
    experiment_dir를 바꿔 Apply를 누르면 checkpoint는 해당 실험 기준으로 다시 선택됩니다.
  </div>
  <div class="summary">
    experiments={len(experiments)} |
    checkpoints={len(checkpoints)} |
    rows={len(rows)} |
    joint={mode_counts['joint']} |
    motion_only={mode_counts['motion_only']} |
    text_only={mode_counts['text_only']}
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>sample</th>
          <th>joint</th>
          <th>motion only</th>
          <th>text only</th>
        </tr>
      </thead>
      <tbody>
        {table_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def create_handler(
    experiments_root: Path,
    default_experiment: str,
    default_checkpoint: str,
    default_variant: str,
):

    class ComparisonHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(experiments_root), **kwargs)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                query = parse_qs(parsed.query)
                requested_experiment_name = query.get(
                    "experiment",
                    [default_experiment],
                )[0]
                experiments = list_experiments(experiments_root)
                selected_experiment = select_experiment(
                    experiments_root=experiments_root,
                    experiments=experiments,
                    requested_experiment=requested_experiment_name,
                    default_experiment=default_experiment,
                )
                selected_experiment_query = experiment_query_value(
                    selected_experiment,
                    experiments_root,
                )
                source_experiment = query.get(
                    "source_experiment",
                    [selected_experiment_query],
                )[0].strip()
                experiment_changed = (
                    bool(source_experiment)
                    and source_experiment != selected_experiment_query
                )

                checkpoints = list_checkpoints(selected_experiment)
                checkpoint_name = query.get("checkpoint", [default_checkpoint])[0]
                if experiment_changed:
                    checkpoint_name = ""
                variant = query.get("variant", [default_variant])[0]
                if variant not in {"overlay", "raw"}:
                    variant = default_variant

                selected_checkpoint = select_checkpoint(
                    checkpoints=checkpoints,
                    requested_checkpoint=checkpoint_name,
                    default_checkpoint=default_checkpoint,
                )

                content = render_page(
                    experiments_root=experiments_root,
                    experiments=experiments,
                    selected_experiment=selected_experiment,
                    selected_checkpoint=selected_checkpoint,
                    checkpoints=checkpoints,
                    variant=variant,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            super().do_GET()

    return ComparisonHandler


def main() -> None:
    args = parse_args()
    default_experiment_input = args.experiment_dir.strip()
    if args.experiments_root.strip():
        experiments_root = Path(args.experiments_root).expanduser().resolve()
    elif default_experiment_input:
        experiments_root = Path(default_experiment_input).expanduser().resolve().parent
    else:
        raise ValueError("Provide either --experiments_root or --experiment_dir.")

    if not experiments_root.exists() or not experiments_root.is_dir():
        raise ValueError(f"Invalid experiments_root: {experiments_root}")

    experiments = list_experiments(experiments_root)
    selected_experiment = select_experiment(
        experiments_root=experiments_root,
        experiments=experiments,
        requested_experiment=default_experiment_input,
        default_experiment="",
    )
    default_experiment = experiment_query_value(selected_experiment, experiments_root)

    checkpoints = list_checkpoints(selected_experiment)
    selected_checkpoint = select_checkpoint(
        checkpoints=checkpoints,
        requested_checkpoint=args.checkpoint.strip(),
        default_checkpoint="",
    )

    handler_cls = create_handler(
        experiments_root=experiments_root,
        default_experiment=default_experiment,
        default_checkpoint=selected_checkpoint.name,
        default_variant=args.variant,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    url = (
        f"http://{args.host}:{args.port}/"
        f"?experiment={quote(default_experiment)}"
        f"&checkpoint={quote(selected_checkpoint.name)}"
        f"&variant={quote(args.variant)}"
    )
    print("=== Track Analysis Comparison Viewer ===")
    print(f"experiments_root: {experiments_root}")
    print(f"default_experiment: {selected_experiment}")
    print(f"default_checkpoint: {selected_checkpoint.name}")
    print(f"default_variant: {args.variant}")
    print(f"open: {url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
