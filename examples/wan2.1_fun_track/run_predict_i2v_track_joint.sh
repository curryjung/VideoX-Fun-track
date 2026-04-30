#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Joint text-motion guidance defaults.
GUIDANCE_MODE="${GUIDANCE_MODE:-joint_tm}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-1.5}"

echo "[trace_exec] run_predict_i2v_track_joint.sh"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE}"
echo "[trace_exec] text_guidance_weight=${TEXT_GUIDANCE_WEIGHT}"
echo "[trace_exec] motion_guidance_weight=${MOTION_GUIDANCE_WEIGHT}"

TRACK_FILE_PATH="${TRACK_FILE_PATH:-}"
METADATA_PATH="${METADATA_PATH:-}"

if [[ -z "${TRACK_FILE_PATH}" && -z "${METADATA_PATH}" ]]; then
  echo "[error] joint_tm requires motion condition input."
  echo "[error] set either TRACK_FILE_PATH=/path/to/*.npz"
  echo "[error] or METADATA_PATH=/path/to/metadata.(json|jsonl|csv) (with TRAIN_DATA_DIR if needed)."
  exit 1
fi

GUIDANCE_MODE="${GUIDANCE_MODE}" \
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
bash examples/wan2.1_fun_track/run_predict_i2v_track.sh "$@"
