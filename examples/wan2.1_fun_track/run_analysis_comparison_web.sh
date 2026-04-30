#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_normalize_fixed_dropout_first-frame_0p1_text_0p1_track_0p1}"  # default selected experiment
CHECKPOINT="${CHECKPOINT:-checkpoint-200}"  # default selected checkpoint
VARIANT="${VARIANT:-overlay}"  # overlay | raw
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8780}"

python scripts/wan2.1_fun_track/analysis_comparison_web.py \
    --experiments_root "${EXPERIMENTS_ROOT}" \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --variant "${VARIANT}" \
    --host "${HOST}" \
    --port "${PORT}"
