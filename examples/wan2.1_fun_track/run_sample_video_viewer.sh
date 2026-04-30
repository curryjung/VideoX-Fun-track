#!/usr/bin/env bash
set -euo pipefail

SAMPLES_DIR="${SAMPLES_DIR:-/data/project-vilab/jaeseok/VideoX-Fun/samples}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

python scripts/sample_video_viewer.py \
    --samples_dir "${SAMPLES_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
