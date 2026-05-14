#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

#
# No-track i2v analysis runner:
# - keeps first-frame conditioning from metadata sample
# - forces track_condition=None in pipeline
# - supports optional text conditioning (USE_TEXT=true/false)
# - keeps original scripts unchanged
#

PYTHON_BIN="${PYTHON_BIN:-python}"

VAL_METADATA_PATH="${VAL_METADATA_PATH:-/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_train_selected64.json}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/data/shared-vilab/datasets/OpenVid-1M}"
TRAIN_DATA_ROOT_MAP_JSON_TRACK="${TRAIN_DATA_ROOT_MAP_JSON_TRACK:-}"
TRAIN_DATA_ROOT_ID_KEY_TRACK="${TRAIN_DATA_ROOT_ID_KEY_TRACK:-root_id}"

SAMPLE_INDEX_LIST_STR="${SAMPLE_INDEX_LIST_STR:-0 1 2 3 4 5 6 7 8 9 10}"
SEED_LIST_STR="${SEED_LIST_STR:-42}"
read -r -a SAMPLE_INDEX_LIST <<< "${SAMPLE_INDEX_LIST_STR}"
read -r -a SEED_LIST <<< "${SEED_LIST_STR}"

# Empty means "use base Wan2.1-Fun i2v model (no finetuned track checkpoint)"
TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH:-}"

GUIDANCE_MODE="${GUIDANCE_MODE:-cfg}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

USE_TEXT="${USE_TEXT:-false}"
USE_PROMPT_FROM_METADATA="${USE_PROMPT_FROM_METADATA:-true}"
PROMPT="${PROMPT:-a high quality natural motion video}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-worst quality, low quality, blurry, static frame}"

TEXT_FEATURE_PATH="${TEXT_FEATURE_PATH:-}"
NEGATIVE_TEXT_FEATURE_PATH="${NEGATIVE_TEXT_FEATURE_PATH:-}"
DEFAULT_UNCOND_TEXT_NPZ="${DEFAULT_UNCOND_TEXT_NPZ:-/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz}"
NULL_TEXT_FALLBACK_PROMPT="${NULL_TEXT_FALLBACK_PROMPT:-a video}"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-4}"

MODEL_TAG="base_wan21_fun_i2v"
if [[ -n "${TRANSFORMER_CHECKPOINT_PATH}" ]]; then
  MODEL_TAG="$(basename "${TRANSFORMER_CHECKPOINT_PATH}")"
fi
SAVE_BASE_DIR="${SAVE_BASE_DIR:-/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-no-track-first-frame/${MODEL_TAG}}"

if [[ "${USE_TEXT}" != "true" && "${USE_TEXT}" != "false" ]]; then
  echo "[error] USE_TEXT must be true or false (got: ${USE_TEXT})"
  exit 1
fi

if [[ "${USE_PROMPT_FROM_METADATA}" != "true" && "${USE_PROMPT_FROM_METADATA}" != "false" ]]; then
  echo "[error] USE_PROMPT_FROM_METADATA must be true or false (got: ${USE_PROMPT_FROM_METADATA})"
  exit 1
fi

if [[ "${USE_TEXT}" == "false" ]]; then
  if [[ -z "${TEXT_FEATURE_PATH}" && ! -f "${DEFAULT_UNCOND_TEXT_NPZ}" ]]; then
    echo "[error] USE_TEXT=false requires DEFAULT_UNCOND_TEXT_NPZ (or TEXT_FEATURE_PATH)."
    echo "[error] missing: ${DEFAULT_UNCOND_TEXT_NPZ}"
    exit 1
  fi
fi

echo "[trace_exec] run_predict_i2v_track_exec_analysis_no_track_first_frame.sh"
echo "[trace_exec] val_metadata_path=${VAL_METADATA_PATH}"
echo "[trace_exec] train_data_dir=${TRAIN_DATA_DIR}"
echo "[trace_exec] sample_index_list=${SAMPLE_INDEX_LIST[*]}"
echo "[trace_exec] seed_list=${SEED_LIST[*]}"
echo "[trace_exec] use_text=${USE_TEXT}"
echo "[trace_exec] use_prompt_from_metadata=${USE_PROMPT_FROM_METADATA}"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE}"
echo "[trace_exec] guidance_scale=${GUIDANCE_SCALE}"
echo "[trace_exec] transformer_checkpoint_path=${TRANSFORMER_CHECKPOINT_PATH:-<base-model>}"
echo "[trace_exec] cuda_visible_devices=${CUDA_DEVICE}"
echo "[trace_exec] save_base_dir=${SAVE_BASE_DIR}"

resolve_first_frame_and_prompt_from_metadata() {
  local metadata_path="$1"
  local sample_index="$2"
  local train_data_dir="$3"
  local root_map_json="$4"
  local root_id_key="$5"

  "${PYTHON_BIN}" - "$metadata_path" "$sample_index" "$train_data_dir" "$root_map_json" "$root_id_key" <<'PY'
import csv
import json
import os
import sys

metadata_path = sys.argv[1]
sample_index = int(sys.argv[2])
train_data_dir = sys.argv[3]
root_map_json = sys.argv[4]
root_id_key = sys.argv[5]

def read_records(path):
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON metadata must be a list: {path}")
        return data
    if lower.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported metadata format: {path}")

records = read_records(metadata_path)
if len(records) == 0:
    raise ValueError(f"Metadata is empty: {metadata_path}")

if sample_index < 0:
    sample_index += len(records)
if sample_index < 0 or sample_index >= len(records):
    raise IndexError(f"sample_index={sample_index} out of range for length={len(records)}")

record = records[sample_index]

root_map = {}
if root_map_json:
    with open(root_map_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"root_map_json must contain object: {root_map_json}")
    root_map = {str(k): str(v) for k, v in loaded.items()}

def resolve_path(path_value):
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    root_id = record.get(root_id_key, None)
    if root_id is not None:
        root_key = str(root_id).strip()
        if root_key in root_map:
            return os.path.join(root_map[root_key], path_value)
    if train_data_dir:
        return os.path.join(train_data_dir, path_value)
    return os.path.abspath(path_value)

first_frame = ""
for key in ("first_frame_path", "image_path", "frame_path"):
    candidate = resolve_path(str(record.get(key, "")).strip())
    if candidate and os.path.isfile(candidate):
        first_frame = candidate
        break

if not first_frame:
    media_path = resolve_path(str(record.get("file_path", "")).strip())
    if media_path and os.path.isfile(media_path):
        lower = media_path.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            first_frame = media_path
    if not first_frame and media_path:
        media_dir = os.path.dirname(media_path)
        for name in (
            "first_frame.png",
            "first_frame.jpg",
            "first_frame.jpeg",
            "first_frame.webp",
            "first_frame.bmp",
        ):
            candidate = os.path.join(media_dir, name)
            if os.path.isfile(candidate):
                first_frame = candidate
                break

if not first_frame:
    raise FileNotFoundError(
        "Could not resolve first frame image from metadata row. "
        f"index={sample_index}, file_path={record.get('file_path', '')}"
    )

prompt = str(record.get("text", "")).replace("\n", " ").strip()
print(sample_index)
print(first_frame)
print(prompt)
PY
}

for SEED in "${SEED_LIST[@]}"; do
  for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
    mapfile -t SAMPLE_INFO < <(
      resolve_first_frame_and_prompt_from_metadata \
        "${VAL_METADATA_PATH}" \
        "${SAMPLE_INDEX}" \
        "${TRAIN_DATA_DIR}" \
        "${TRAIN_DATA_ROOT_MAP_JSON_TRACK}" \
        "${TRAIN_DATA_ROOT_ID_KEY_TRACK}"
    )

    if [[ "${#SAMPLE_INFO[@]}" -lt 2 ]]; then
      echo "[error] failed to resolve metadata sample info for sample_index=${SAMPLE_INDEX}"
      exit 1
    fi

    RESOLVED_INDEX="${SAMPLE_INFO[0]}"
    VALIDATION_IMAGE_START="${SAMPLE_INFO[1]}"
    METADATA_PROMPT="${SAMPLE_INFO[2]:-}"

    RUN_PROMPT="${PROMPT}"
    RUN_TEXT_FEATURE_PATH="${TEXT_FEATURE_PATH}"
    RUN_NEGATIVE_TEXT_FEATURE_PATH="${NEGATIVE_TEXT_FEATURE_PATH}"
    TEXT_TAG="text_on"

    if [[ "${USE_TEXT}" == "true" ]]; then
      if [[ "${USE_PROMPT_FROM_METADATA}" == "true" && -n "${METADATA_PROMPT}" ]]; then
        RUN_PROMPT="${METADATA_PROMPT}"
      fi
      if [[ -z "${RUN_PROMPT// }" && -z "${RUN_TEXT_FEATURE_PATH}" ]]; then
        RUN_PROMPT="a video"
      fi
    else
      TEXT_TAG="text_off"
      RUN_PROMPT="${NULL_TEXT_FALLBACK_PROMPT}"
      if [[ -z "${RUN_TEXT_FEATURE_PATH}" ]]; then
        RUN_TEXT_FEATURE_PATH="${DEFAULT_UNCOND_TEXT_NPZ}"
      fi
      if [[ -z "${RUN_NEGATIVE_TEXT_FEATURE_PATH}" ]]; then
        RUN_NEGATIVE_TEXT_FEATURE_PATH="${RUN_TEXT_FEATURE_PATH}"
      fi
    fi

    SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${RESOLVED_INDEX}/${TEXT_TAG}"
    mkdir -p "${SAVE_DIR}"

    OUTPUT_NAME_SUFFIX="notrack_firstframe_${TEXT_TAG}_seed${SEED}"
    echo "[trace_exec] sample_index=${SAMPLE_INDEX} resolved_index=${RESOLVED_INDEX} seed=${SEED}"
    echo "[trace_exec] validation_image_start=${VALIDATION_IMAGE_START}"
    echo "[trace_exec] output_name_suffix=${OUTPUT_NAME_SUFFIX}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
    VALIDATION_IMAGE_START="${VALIDATION_IMAGE_START}" \
    PROMPT="${RUN_PROMPT}" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
    TEXT_FEATURE_PATH="${RUN_TEXT_FEATURE_PATH}" \
    NEGATIVE_TEXT_FEATURE_PATH="${RUN_NEGATIVE_TEXT_FEATURE_PATH}" \
    TRACK_FILE_PATH="" \
    METADATA_PATH="" \
    TRAIN_DATA_DIR="" \
    USE_PROMPT_FROM_METADATA=false \
    FORCE_TRACK_CONDITION_NONE=true \
    GUIDANCE_MODE="${GUIDANCE_MODE}" \
    GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    TRACK_ANALYSIS=false \
    DEBUG_TRACK_CONDITION=false \
    SAVE_DIR="${SAVE_DIR}" \
    OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX}" \
    SEED="${SEED}" \
    bash examples/wan2.1_fun_track/run_predict_i2v_track.sh
  done
done
