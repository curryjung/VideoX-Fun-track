#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Motion-only guidance with random track-point sampling to mirror train-time augmentation.
GUIDANCE_MODE="${GUIDANCE_MODE:-motion_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-0.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.5}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-random}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-false}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-}"

echo "[trace_exec] run_predict_i2v_track_motion_only_random.sh"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE}"
echo "[trace_exec] text_guidance_weight=${TEXT_GUIDANCE_WEIGHT}"
echo "[trace_exec] motion_guidance_weight=${MOTION_GUIDANCE_WEIGHT}"
echo "[trace_exec] track_point_sample_mode=${TRACK_POINT_SAMPLE_MODE}"
echo "[trace_exec] track_sort_selected_indices=${TRACK_SORT_SELECTED_INDICES}"
echo "[trace_exec] track_point_sample_seed=${TRACK_POINT_SAMPLE_SEED:-<random>}"

GUIDANCE_MODE="${GUIDANCE_MODE}" \
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE}" \
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES}" \
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED}" \
bash examples/wan2.1_fun_track/run_predict_i2v_track_motion_only.sh "$@"
