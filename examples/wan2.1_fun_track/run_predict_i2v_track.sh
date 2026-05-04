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
VALIDATION_IMAGE_START="${VALIDATION_IMAGE_START:-asset/1.png}"
VALIDATION_IMAGE_END="${VALIDATION_IMAGE_END:-}"
TRACK_FILE_PATH="${TRACK_FILE_PATH:-}"
# Set TRACK_NORMALIZE=true when track npz stores pixel coords that should match
# the training --track_normalize path (pixel -> [0,1] before model-side canvas mapping).
# If a track npz is already [0,1] normalized, do not enable this without also
# adapting the loader; otherwise the coordinates will be divided twice.
TRACK_NORMALIZE="${TRACK_NORMALIZE:-true}"
OVERLAY_LINEWIDTH="${OVERLAY_LINEWIDTH:-2}"
OVERLAY_TRACE_FRAMES="${OVERLAY_TRACE_FRAMES:--1}"
OVERLAY_PAD_VALUE="${OVERLAY_PAD_VALUE:-0}"
COTRACKER_ROOT="${COTRACKER_ROOT:-/data/project-vilab/jaeseok/co-tracker}"
METADATA_PATH="${METADATA_PATH:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
RANDOM_SAMPLE="${RANDOM_SAMPLE:-false}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-}"
TRAIN_DATA_ROOT_MAP_JSON_TRACK="${TRAIN_DATA_ROOT_MAP_JSON_TRACK:-}"
TRAIN_DATA_ROOT_ID_KEY_TRACK="${TRAIN_DATA_ROOT_ID_KEY_TRACK:-root_id}"
USE_PROMPT_FROM_METADATA="${USE_PROMPT_FROM_METADATA:-true}"
TEXT_FEATURE_PATH="${TEXT_FEATURE_PATH:-}"
NEGATIVE_TEXT_FEATURE_PATH="${NEGATIVE_TEXT_FEATURE_PATH:-}"
CLIP_FEATURE_PATH="${CLIP_FEATURE_PATH:-}"
ZERO_CLIP_CONTEXT="${ZERO_CLIP_CONTEXT:-false}"
GUIDANCE_MODE="${GUIDANCE_MODE:-cfg}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-1.5}"
DEFAULT_UNCOND_TEXT_NPZ="${DEFAULT_UNCOND_TEXT_NPZ:-/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz}"
# DEBUG_TRACK_CONDITION=true → print tensors only (no pdb).
# TRACK_ANALYSIS=true → print quantitative analysis (concat norm / patch diff / block1-2 retention at step 0).
# PDB_TRACK_CONDITION=true → stop once in pdb before pipeline() (needs interactive TTY).
# PDB_PIPELINE_STEP0=true → stop in pdb at first denoise step before transformer forward.
DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-false}"
TRACK_ANALYSIS="${TRACK_ANALYSIS:-false}"
PDB_TRACK_CONDITION="${PDB_TRACK_CONDITION:-false}"
PDB_PIPELINE_STEP0="${PDB_PIPELINE_STEP0:-false}"
FORCE_TRACK_CONDITION_NONE="${FORCE_TRACK_CONDITION_NONE:-false}"
RANDOM_FAKE_TRACK="${RANDOM_FAKE_TRACK:-false}"
TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE:-1.0}"
TRACK_LATENT_FIRST_FRAME_SCALE="${TRACK_LATENT_FIRST_FRAME_SCALE:-}"
TRACK_LATENT_REST_FRAME_SCALE="${TRACK_LATENT_REST_FRAME_SCALE:-}"
TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM:-}"
TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET:-0}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-uniform}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-true}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-}"
TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE:-original}"
SEED="${SEED:-42}"

SAVE_DIR="${SAVE_DIR:-samples/wan-videos-fun-i2v-track}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-}"

if [[ "${GUIDANCE_MODE}" != "cfg" && -z "${NEGATIVE_TEXT_FEATURE_PATH}" ]]; then
  if [[ -f "${DEFAULT_UNCOND_TEXT_NPZ}" ]]; then
    NEGATIVE_TEXT_FEATURE_PATH="${DEFAULT_UNCOND_TEXT_NPZ}"
  else
    echo "[warn] custom guidance requested but DEFAULT_UNCOND_TEXT_NPZ not found: ${DEFAULT_UNCOND_TEXT_NPZ}"
  fi
fi

if [[ -z "${OUTPUT_NAME_SUFFIX}" ]]; then
  TRACK_MODE_SUFFIX="track_on"
  if [[ "${FORCE_TRACK_CONDITION_NONE}" == "true" ]]; then
    TRACK_MODE_SUFFIX="track_off"
  elif [[ "${RANDOM_FAKE_TRACK}" == "true" ]]; then
    TRACK_MODE_SUFFIX="track_fake"
  fi
  TRACK_POINTS_SUFFIX="pall"
  if [[ -n "${TRACK_MAX_POINTS:-}" && "${TRACK_MAX_POINTS}" != "-1" ]]; then
    TRACK_POINTS_SUFFIX="p${TRACK_MAX_POINTS}"
  fi
  GUIDANCE_SUFFIX=""
  if [[ "${GUIDANCE_MODE}" == "joint_tm" ]]; then
    GUIDANCE_SUFFIX="_jointtm_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}"
  elif [[ "${GUIDANCE_MODE}" == "text_only" ]]; then
    GUIDANCE_SUFFIX="_textonly_wt${TEXT_GUIDANCE_WEIGHT}"
  elif [[ "${GUIDANCE_MODE}" == "motion_only" ]]; then
    GUIDANCE_SUFFIX="_motiononly_wm${MOTION_GUIDANCE_WEIGHT}"
  elif [[ "${GUIDANCE_MODE}" == "unified" ]]; then
    GUIDANCE_SUFFIX="_unified_gs${GUIDANCE_SCALE}"
  fi
  TRACK_OFFSET_SUFFIX=""
  if [[ "${TRACK_CONDITION_INDEX_OFFSET}" != "0" ]]; then
    TRACK_OFFSET_SUFFIX="_toff${TRACK_CONDITION_INDEX_OFFSET}"
  fi
  OUTPUT_NAME_SUFFIX="${TRACK_MODE_SUFFIX}_${TRACK_POINTS_SUFFIX}${GUIDANCE_SUFFIX}${TRACK_OFFSET_SUFFIX}"
fi

echo "[trace_exec] run_predict_i2v_track.sh"
echo "[trace_exec] repo_root=${REPO_ROOT}"
echo "[trace_exec] python_bin=${PYTHON_BIN}"
echo "[trace_exec] validation_image_start=${VALIDATION_IMAGE_START}"
echo "[trace_exec] track_file_path=${TRACK_FILE_PATH:-<empty>}"
echo "[trace_exec] text_feature_path=${TEXT_FEATURE_PATH:-<empty>}"
echo "[trace_exec] negative_text_feature_path=${NEGATIVE_TEXT_FEATURE_PATH:-<empty>}"
echo "[trace_exec] clip_feature_path=${CLIP_FEATURE_PATH:-<empty>}"
echo "[trace_exec] zero_clip_context=${ZERO_CLIP_CONTEXT}"
echo "[trace_exec] metadata_path=${METADATA_PATH:-<empty>}"
echo "[trace_exec] guidance_mode=${GUIDANCE_MODE} guidance_scale=${GUIDANCE_SCALE} text_guidance_weight=${TEXT_GUIDANCE_WEIGHT} motion_guidance_weight=${MOTION_GUIDANCE_WEIGHT}"
echo "[trace_exec] debug_track_condition=${DEBUG_TRACK_CONDITION} pdb_track_condition=${PDB_TRACK_CONDITION} pdb_pipeline_step0=${PDB_PIPELINE_STEP0}"
echo "[trace_exec] track_analysis=${TRACK_ANALYSIS}"
echo "[trace_exec] force_track_condition_none=${FORCE_TRACK_CONDITION_NONE}"
echo "[trace_exec] random_fake_track=${RANDOM_FAKE_TRACK}"
echo "[trace_exec] track_latent_scale=${TRACK_LATENT_SCALE}"
echo "[trace_exec] track_latent_first_frame_scale=${TRACK_LATENT_FIRST_FRAME_SCALE:-<track_latent_scale>}"
echo "[trace_exec] track_latent_rest_frame_scale=${TRACK_LATENT_REST_FRAME_SCALE:-<track_latent_scale>}"
echo "[trace_exec] track_head_hidden_dim=${TRACK_HEAD_HIDDEN_DIM:-<config>}"
echo "[trace_exec] track_condition_index_offset=${TRACK_CONDITION_INDEX_OFFSET}"
echo "[trace_exec] track_point_sample_mode=${TRACK_POINT_SAMPLE_MODE}"
echo "[trace_exec] track_sort_selected_indices=${TRACK_SORT_SELECTED_INDICES}"
echo "[trace_exec] track_point_sample_seed=${TRACK_POINT_SAMPLE_SEED:-<default:seed>}"
echo "[trace_exec] track_point_id_mode=${TRACK_POINT_ID_MODE}"
echo "[trace_exec] seed=${SEED}"
echo "[trace_exec] output_name_suffix=${OUTPUT_NAME_SUFFIX}"

EXTRA_ARGS=()
if [[ -n "${TRANSFORMER_CHECKPOINT_PATH}" ]]; then
  EXTRA_ARGS+=("--transformer_checkpoint_path=${TRANSFORMER_CHECKPOINT_PATH}")
fi
if [[ -n "${VALIDATION_IMAGE_END}" ]]; then
  EXTRA_ARGS+=("--validation_image_end=${VALIDATION_IMAGE_END}")
fi
if [[ -n "${VALIDATION_IMAGE_START}" ]]; then
  EXTRA_ARGS+=("--validation_image_start=${VALIDATION_IMAGE_START}")
fi
if [[ -n "${TRACK_FILE_PATH}" ]]; then
  EXTRA_ARGS+=("--track_file_path=${TRACK_FILE_PATH}")
fi
if [[ -n "${TRACK_MAX_POINTS:-}" ]]; then
  EXTRA_ARGS+=("--track_max_points=${TRACK_MAX_POINTS}")
fi
if [[ -n "${TRACK_POINT_SAMPLE_MODE}" ]]; then
  EXTRA_ARGS+=("--track_point_sample_mode=${TRACK_POINT_SAMPLE_MODE}")
fi
if [[ -n "${TRACK_SORT_SELECTED_INDICES}" ]]; then
  EXTRA_ARGS+=("--track_sort_selected_indices=${TRACK_SORT_SELECTED_INDICES}")
fi
if [[ -n "${TRACK_POINT_SAMPLE_SEED}" ]]; then
  EXTRA_ARGS+=("--track_point_sample_seed=${TRACK_POINT_SAMPLE_SEED}")
fi
if [[ -n "${TRACK_POINT_ID_MODE}" ]]; then
  EXTRA_ARGS+=("--track_point_id_mode=${TRACK_POINT_ID_MODE}")
fi
if [[ -n "${SEED}" ]]; then
  EXTRA_ARGS+=("--seed=${SEED}")
fi
if [[ "${TRACK_NORMALIZE}" == "true" ]]; then
  EXTRA_ARGS+=("--normalize_track")
fi
if [[ -n "${METADATA_PATH}" ]]; then
  EXTRA_ARGS+=("--metadata_path=${METADATA_PATH}" "--sample_index=${SAMPLE_INDEX}")
fi
if [[ "${RANDOM_SAMPLE}" == "true" ]]; then
  EXTRA_ARGS+=("--random_sample")
fi
if [[ -n "${TRAIN_DATA_DIR}" ]]; then
  EXTRA_ARGS+=("--train_data_dir=${TRAIN_DATA_DIR}")
fi
if [[ -n "${TRAIN_DATA_ROOT_MAP_JSON_TRACK}" ]]; then
  EXTRA_ARGS+=("--train_data_root_map_json_track=${TRAIN_DATA_ROOT_MAP_JSON_TRACK}")
fi
if [[ -n "${TRAIN_DATA_ROOT_ID_KEY_TRACK}" ]]; then
  EXTRA_ARGS+=("--train_data_root_id_key_track=${TRAIN_DATA_ROOT_ID_KEY_TRACK}")
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
if [[ -n "${CLIP_FEATURE_PATH}" ]]; then
  EXTRA_ARGS+=("--clip_feature_path=${CLIP_FEATURE_PATH}")
fi
if [[ "${ZERO_CLIP_CONTEXT}" == "true" ]]; then
  EXTRA_ARGS+=("--zero_clip_context")
fi
EXTRA_ARGS+=("--overlay_linewidth=${OVERLAY_LINEWIDTH}" "--overlay_trace_frames=${OVERLAY_TRACE_FRAMES}")
if [[ -n "${COTRACKER_ROOT}" ]]; then
  EXTRA_ARGS+=("--cotracker_root=${COTRACKER_ROOT}")
fi
EXTRA_ARGS+=("--overlay_pad_value=${OVERLAY_PAD_VALUE}")
if [[ "${DEBUG_TRACK_CONDITION}" == "true" ]]; then
  EXTRA_ARGS+=("--debug_track_condition")
fi
if [[ "${TRACK_ANALYSIS}" == "true" ]]; then
  EXTRA_ARGS+=("--track_analysis")
fi
if [[ "${PDB_TRACK_CONDITION}" == "true" ]]; then
  EXTRA_ARGS+=("--pdb_track_condition")
fi
if [[ "${PDB_PIPELINE_STEP0}" == "true" ]]; then
  EXTRA_ARGS+=("--pdb_pipeline_step0")
fi
if [[ "${FORCE_TRACK_CONDITION_NONE}" == "true" ]]; then
  EXTRA_ARGS+=("--force_track_condition_none")
fi
if [[ "${RANDOM_FAKE_TRACK}" == "true" ]]; then
  EXTRA_ARGS+=("--random_fake_track")
fi
if [[ -n "${TRACK_LATENT_SCALE}" ]]; then
  EXTRA_ARGS+=("--track_latent_scale=${TRACK_LATENT_SCALE}")
fi
if [[ -n "${TRACK_LATENT_FIRST_FRAME_SCALE}" ]]; then
  export TRACK_LATENT_FIRST_FRAME_SCALE
fi
if [[ -n "${TRACK_LATENT_REST_FRAME_SCALE}" ]]; then
  export TRACK_LATENT_REST_FRAME_SCALE
fi
if [[ -n "${TRACK_HEAD_HIDDEN_DIM}" ]]; then
  EXTRA_ARGS+=("--track_head_hidden_dim=${TRACK_HEAD_HIDDEN_DIM}")
fi
if [[ -n "${TRACK_CONDITION_INDEX_OFFSET}" ]]; then
  EXTRA_ARGS+=("--track_condition_index_offset=${TRACK_CONDITION_INDEX_OFFSET}")
fi
if [[ -n "${GUIDANCE_MODE}" ]]; then
  EXTRA_ARGS+=("--guidance_mode=${GUIDANCE_MODE}")
fi
if [[ -n "${GUIDANCE_SCALE}" ]]; then
  EXTRA_ARGS+=("--guidance_scale=${GUIDANCE_SCALE}")
fi
if [[ -n "${TEXT_GUIDANCE_WEIGHT}" ]]; then
  EXTRA_ARGS+=("--text_guidance_weight=${TEXT_GUIDANCE_WEIGHT}")
fi
if [[ -n "${MOTION_GUIDANCE_WEIGHT}" ]]; then
  EXTRA_ARGS+=("--motion_guidance_weight=${MOTION_GUIDANCE_WEIGHT}")
fi

printf '[trace_exec] python argv extras:'
for arg in "${EXTRA_ARGS[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${PYTHON_BIN}" examples/wan2.1_fun_track/predict_i2v_track.py \
  --config_path="${CONFIG_PATH}" \
  --model_name="${MODEL_NAME}" \
  --prompt="${PROMPT}" \
  --negative_prompt="${NEGATIVE_PROMPT}" \
  --save_dir="${SAVE_DIR}" \
  --output_name_suffix="${OUTPUT_NAME_SUFFIX}" \
  "${EXTRA_ARGS[@]}"
