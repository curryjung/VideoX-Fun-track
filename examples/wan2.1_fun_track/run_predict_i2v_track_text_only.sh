#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Text-only guidance defaults (motion branch is always null).
GUIDANCE_MODE="${GUIDANCE_MODE:-text_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-0.0}"

echo "[trace_exec] run_predict_i2v_track_text_only.sh"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE}"
echo "[trace_exec] text_guidance_weight=${TEXT_GUIDANCE_WEIGHT}"
echo "[trace_exec] motion_guidance_weight=${MOTION_GUIDANCE_WEIGHT}"

GUIDANCE_MODE="${GUIDANCE_MODE}" \
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
bash examples/wan2.1_fun_track/run_predict_i2v_track.sh "$@"
