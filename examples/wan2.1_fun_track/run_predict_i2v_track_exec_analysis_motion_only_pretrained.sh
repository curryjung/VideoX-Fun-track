#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Pretrained track-model analysis runner (motion-only guidance).
# - Existing script is kept unchanged.
# - This script targets a single pretrained checkpoint path.
# - Track analysis is enabled by default so patch-group contribution metrics are logged.

PYTHON_BIN="${PYTHON_BIN:-python}"

VAL_METADATA_PATH="${VAL_METADATA_PATH:-/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_bin8_train_78k.json}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/data/shared-vilab/datasets/OpenVid-1M}"

PRETRAINED_TRACK_CKPT="${PRETRAINED_TRACK_CKPT:-${TRANSFORMER_CHECKPOINT_PATH:-/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track_local-point-id-mode_bs64_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-2200}}"
if [[ -z "${PRETRAINED_TRACK_CKPT}" ]]; then
  echo "[error] PRETRAINED_TRACK_CKPT (or TRANSFORMER_CHECKPOINT_PATH) is required."
  echo "[error] example:"
  echo "        PRETRAINED_TRACK_CKPT=/path/to/checkpoint-5600 bash $0"
  exit 1
fi

SAMPLE_INDEX_LIST_STR="${SAMPLE_INDEX_LIST_STR:-0}"
SEED_LIST_STR="${SEED_LIST_STR:-42}"
TRACK_LATENT_SCALE_LIST_STR="${TRACK_LATENT_SCALE_LIST_STR:-1.0}"

read -r -a SAMPLE_INDEX_LIST <<< "${SAMPLE_INDEX_LIST_STR}"
read -r -a SEED_LIST <<< "${SEED_LIST_STR}"
read -r -a TRACK_LATENT_SCALE_LIST <<< "${TRACK_LATENT_SCALE_LIST_STR}"

GUIDANCE_MODE="${GUIDANCE_MODE:-motion_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-0.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.0}"
TRACK_NORMALIZE="${TRACK_NORMALIZE:-true}"
TRACK_NORMALIZE_HEIGHT="${TRACK_NORMALIZE_HEIGHT:-480}"
TRACK_NORMALIZE_WIDTH="${TRACK_NORMALIZE_WIDTH:-832}"
TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM:-64}"
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-2000}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-random}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-false}"
TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE:-local}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-42}"
TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET:-0}"

DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-false}"
TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"

VISUALIZE_TRACK_NORMALIZATION="${VISUALIZE_TRACK_NORMALIZATION:-true}"
VIS_TRACK_MODES="${VIS_TRACK_MODES:-auto normalized pixel}"
VIS_TRACK_SAVE_MP4="${VIS_TRACK_SAVE_MP4:-false}"
VIS_TRACK_MAX_FRAMES="${VIS_TRACK_MAX_FRAMES:-32}"
VIS_TRACK_FPS="${VIS_TRACK_FPS:-8}"
VIS_TRACK_HEAT_PERCENTILE="${VIS_TRACK_HEAT_PERCENTILE:-99.0}"
VIS_TRACK_HEAT_GAMMA="${VIS_TRACK_HEAT_GAMMA:-0.55}"

CKPT_PARENT_NAME="$(basename "$(dirname "${PRETRAINED_TRACK_CKPT}")")"
CKPT_NAME="$(basename "${PRETRAINED_TRACK_CKPT}")"
SAVE_BASE_DIR_BASE="${SAVE_BASE_DIR_BASE:-/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track}"
SAVE_BASE_DIR="${SAVE_BASE_DIR_BASE}/${CKPT_PARENT_NAME}/${CKPT_NAME}/analysis_from_val_metadata_motion_only_pretrained"

echo "[trace_exec] run_predict_i2v_track_exec_analysis_motion_only_pretrained.sh"
echo "[trace_exec] pretrained_track_ckpt=${PRETRAINED_TRACK_CKPT}"
echo "[trace_exec] val_metadata_path=${VAL_METADATA_PATH}"
echo "[trace_exec] train_data_dir=${TRAIN_DATA_DIR}"
echo "[trace_exec] sample_index_list=${SAMPLE_INDEX_LIST[*]}"
echo "[trace_exec] seed_list=${SEED_LIST[*]}"
echo "[trace_exec] track_latent_scale_list=${TRACK_LATENT_SCALE_LIST[*]}"
echo "[trace_exec] track_analysis=${TRACK_ANALYSIS}"

for SEED in "${SEED_LIST[@]}"; do
  for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
    for TRACK_LATENT_SCALE in "${TRACK_LATENT_SCALE_LIST[@]}"; do
      SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${SAMPLE_INDEX}/scale_${TRACK_LATENT_SCALE}"
      mkdir -p "${SAVE_DIR}"

      TRACK_POINTS_SUFFIX="pall"
      if [[ "${TRACK_MAX_POINTS}" != "-1" ]]; then
        TRACK_POINTS_SUFFIX="p${TRACK_MAX_POINTS}"
      fi
      OUTPUT_NAME_SUFFIX="motiononly_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_${TRACK_POINTS_SUFFIX}_s${TRACK_LATENT_SCALE}_toff${TRACK_CONDITION_INDEX_OFFSET}_seed${SEED}"

      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      TRANSFORMER_CHECKPOINT_PATH="${PRETRAINED_TRACK_CKPT}" \
      METADATA_PATH="${VAL_METADATA_PATH}" \
      TRAIN_DATA_DIR="${TRAIN_DATA_DIR}" \
      SAMPLE_INDEX="${SAMPLE_INDEX}" \
      USE_PROMPT_FROM_METADATA=true \
      COTRACKER_ROOT="${COTRACKER_ROOT:-/data/project-vilab/jaeseok/co-tracker}" \
      SAVE_DIR="${SAVE_DIR}" \
      OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX}" \
      OVERLAY_PAD_VALUE=0 \
      OVERLAY_LINEWIDTH=1 \
      OVERLAY_TRACE_FRAMES=8 \
      TRACK_MAX_POINTS="${TRACK_MAX_POINTS}" \
      TRACK_NORMALIZE="${TRACK_NORMALIZE}" \
      TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM}" \
      TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE}" \
      TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET}" \
      TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE}" \
      TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES}" \
      TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE}" \
      TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED}" \
      SEED="${SEED}" \
      DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
      TRACK_ANALYSIS="${TRACK_ANALYSIS}" \
      GUIDANCE_MODE="${GUIDANCE_MODE}" \
      TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
      MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
      bash examples/wan2.1_fun_track/run_predict_i2v_track_motion_only.sh

      if [[ "${VISUALIZE_TRACK_NORMALIZATION}" == "true" ]]; then
        VIS_SAVE_DIR="${SAVE_DIR}/track_canvas_debug"
        read -r -a VIS_TRACK_MODE_ARGS <<< "${VIS_TRACK_MODES}"
        VIS_ARGS=(
          --meta_json "${VAL_METADATA_PATH}"
          --data_root "${TRAIN_DATA_DIR}"
          --meta_offset "${SAMPLE_INDEX}"
          --meta_count 1
          --output_dir "${VIS_SAVE_DIR}"
          --latent_h 60
          --latent_w 104
          --track_resolution_h "${TRACK_NORMALIZE_HEIGHT}"
          --track_resolution_w "${TRACK_NORMALIZE_WIDTH}"
          --modes "${VIS_TRACK_MODE_ARGS[@]}"
          --frame_index 0
          --heat_percentile "${VIS_TRACK_HEAT_PERCENTILE}"
          --heat_gamma "${VIS_TRACK_HEAT_GAMMA}"
        )
        if [[ "${TRACK_NORMALIZE}" == "true" ]]; then
          VIS_ARGS+=(--apply_track_normalize)
        fi
        if [[ "${VIS_TRACK_SAVE_MP4}" == "true" ]]; then
          VIS_ARGS+=(
            --save_mp4
            --max_frames "${VIS_TRACK_MAX_FRAMES}"
            --fps "${VIS_TRACK_FPS}"
          )
        fi
        echo "[track_norm_vis] sample_index=${SAMPLE_INDEX} save_dir=${VIS_SAVE_DIR}"
        "${PYTHON_BIN}" scripts/wan2.1_fun_track/visualize_track_canvas.py "${VIS_ARGS[@]}"
      fi
    done
  done
done
