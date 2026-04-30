#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_NAME="${MODEL_NAME:-models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP}"
CONFIG_PATH="${CONFIG_PATH:-config/wan2.1/wan_civitai.yaml}"
TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH:-}"

PROMPT="${PROMPT:-a woman walking in a park, cinematic lighting}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-worst quality, low quality, blurry, static frame}"
METADATA_PATH="${METADATA_PATH:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-false}"
USE_PROMPT_FROM_METADATA="${USE_PROMPT_FROM_METADATA:-true}"

TEXT_FEATURE_PATH="${TEXT_FEATURE_PATH:-}"
NEGATIVE_TEXT_FEATURE_PATH="${NEGATIVE_TEXT_FEATURE_PATH:-}"

SAMPLE_HEIGHT="${SAMPLE_HEIGHT:-480}"
SAMPLE_WIDTH="${SAMPLE_WIDTH:-832}"
VIDEO_LENGTH="${VIDEO_LENGTH:-81}"
FPS="${FPS:-16}"
GUIDANCE_MODE="${GUIDANCE_MODE:-cfg}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-42}"
SAMPLER_NAME="${SAMPLER_NAME:-Flow}"
SHIFT="${SHIFT:-3.0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

SAVE_DIR="${SAVE_DIR:-samples/wan-videos-fun-i2v-track-no-first-frame}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-noff_notrack}"

if [[ -n "${VALIDATION_IMAGE_START:-}" ]]; then
  echo "[warn] VALIDATION_IMAGE_START is ignored in no-first-frame mode."
fi
if [[ -n "${TRACK_FILE_PATH:-}" ]]; then
  echo "[warn] TRACK_FILE_PATH is ignored in no-track mode."
fi
if [[ -n "${CLIP_FEATURE_PATH:-}" ]]; then
  echo "[warn] CLIP_FEATURE_PATH is ignored in no-first-frame mode."
fi

echo "[trace_exec] run_predict_i2v_track_no_first_frame.sh"
echo "[trace_exec] repo_root=${REPO_ROOT}"
echo "[trace_exec] python_bin=${PYTHON_BIN}"
echo "[trace_exec] no_first_frame=true no_track_condition=true"
echo "[trace_exec] metadata_path=${METADATA_PATH:-<empty>}"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE}"

EXTRA_ARGS=(
  "--sample_height=${SAMPLE_HEIGHT}"
  "--sample_width=${SAMPLE_WIDTH}"
  "--video_length=${VIDEO_LENGTH}"
  "--fps=${FPS}"
  "--guidance_mode=${GUIDANCE_MODE}"
  "--guidance_scale=${GUIDANCE_SCALE}"
  "--text_guidance_weight=${TEXT_GUIDANCE_WEIGHT}"
  "--num_inference_steps=${NUM_INFERENCE_STEPS}"
  "--seed=${SEED}"
  "--sampler_name=${SAMPLER_NAME}"
  "--shift=${SHIFT}"
  "--mixed_precision=${MIXED_PRECISION}"
)

if [[ -n "${TRANSFORMER_CHECKPOINT_PATH}" ]]; then
  EXTRA_ARGS+=("--transformer_checkpoint_path=${TRANSFORMER_CHECKPOINT_PATH}")
fi
if [[ -n "${METADATA_PATH}" ]]; then
  EXTRA_ARGS+=("--metadata_path=${METADATA_PATH}" "--sample_index=${SAMPLE_INDEX}")
fi
if [[ "${RANDOM_SAMPLE}" == "true" ]]; then
  EXTRA_ARGS+=("--random_sample")
fi
if [[ "${USE_PROMPT_FROM_METADATA}" == "true" ]]; then
  EXTRA_ARGS+=("--use_prompt_from_metadata")
fi
if [[ -n "${TEXT_FEATURE_PATH}" ]]; then
  EXTRA_ARGS+=("--text_feature_path=${TEXT_FEATURE_PATH}")
fi
if [[ -n "${NEGATIVE_TEXT_FEATURE_PATH}" ]]; then
  EXTRA_ARGS+=("--negative_text_feature_path=${NEGATIVE_TEXT_FEATURE_PATH}")
fi
if [[ -n "${OUTPUT_NAME_SUFFIX}" ]]; then
  EXTRA_ARGS+=("--output_name_suffix=${OUTPUT_NAME_SUFFIX}")
fi

printf '[trace_exec] python argv extras:'
for arg in "${EXTRA_ARGS[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${PYTHON_BIN}" examples/wan2.1_fun_track/predict_i2v_track_no_first_frame.py \
  --config_path="${CONFIG_PATH}" \
  --model_name="${MODEL_NAME}" \
  --prompt="${PROMPT}" \
  --negative_prompt="${NEGATIVE_PROMPT}" \
  --save_dir="${SAVE_DIR}" \
  "${EXTRA_ARGS[@]}"
