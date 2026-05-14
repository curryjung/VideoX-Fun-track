#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv-videoxfun/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv-videoxfun/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"

DAVIS_ROOT="${DAVIS_ROOT:-/data/project-vilab/jaeseok/davis/DAVIS/JPEGImages/480p}"
MODEL_NAME="${MODEL_NAME:-models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP}"
CONFIG_PATH="${CONFIG_PATH:-config/wan2.1/wan_civitai.yaml}"
NEGATIVE_TEXT_FEATURE_PATH="${NEGATIVE_TEXT_FEATURE_PATH:-/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/data/project-vilab/jaeseok/VideoX-Fun/checkpoints}"
EXP_NAME="${EXP_NAME:-wan_track_wan_move_condition_bin8_train_78k_dropout_first-frame_0p1_text_0p1_track_0p1}"
CKPT_LIST="${CKPT_LIST:-11800}"
read -r -a ckpt_list <<< "${CKPT_LIST}"

SAVE_BASE_DIR_BASE="${SAVE_BASE_DIR_BASE:-/data/project-vilab/jaeseok/VideoX-Fun/evaluation/results}"
GT_TRACK_CACHE_DIR="${GT_TRACK_CACHE_DIR:-${SAVE_BASE_DIR_BASE}/davis_gt_track_cache}"
OVERWRITE_GT_TRACK_CACHE="${OVERWRITE_GT_TRACK_CACHE:-false}"

MAX_VIDEOS="${MAX_VIDEOS:-20}"
VIDEO_LENGTH="${VIDEO_LENGTH:-81}"
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"

GUIDANCE_MODE="${GUIDANCE_MODE:-motion_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-1.5}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"

TRACK_CONDITION_MODE="wan_move"
# <=0 lets the python side use the VAE temporal compression ratio. This matches
# train_track.py, which used vae.config.temporal_compression_ratio for wan_move.
WAN_MOVE_TEMPORAL_STRIDE="${WAN_MOVE_TEMPORAL_STRIDE:-0}"
TRACK_NORMALIZE="${TRACK_NORMALIZE:-true}"
TRACK_NORMALIZE_HEIGHT="${TRACK_NORMALIZE_HEIGHT:-480}"
TRACK_NORMALIZE_WIDTH="${TRACK_NORMALIZE_WIDTH:-832}"

# The training launcher sampled 1..200 track points and used original point ids.
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-1500}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-random}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-true}"
TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE:-original}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-42}"

TRACK_OVERLAY_TRACE_FRAMES="${TRACK_OVERLAY_TRACE_FRAMES:-8}"
TRACK_OVERLAY_SCALE="${TRACK_OVERLAY_SCALE:-0.5}"
TRACK_OVERLAY_CRF="${TRACK_OVERLAY_CRF:-16}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-}"

METRICS="${METRICS:-epe psnr ssim lpips}"
REUSE_GENERATED_FRAMES="${REUSE_GENERATED_FRAMES:-never}"
LPIPS_NET="${LPIPS_NET:-alex}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-16}"
LPIPS_DEVICE="${LPIPS_DEVICE:-}"
read -r -a METRIC_ARGS <<< "${METRICS}"

if [[ "${#ckpt_list[@]}" -eq 0 ]]; then
  echo "[davis_eval_wan_move] empty CKPT_LIST"
  exit 1
fi

guidance_mode_suffix="${GUIDANCE_MODE}"
if [[ "${GUIDANCE_MODE}" == "motion_only" ]]; then
  guidance_mode_suffix="motiononly"
elif [[ "${GUIDANCE_MODE}" == "text_only" ]]; then
  guidance_mode_suffix="textonly"
elif [[ "${GUIDANCE_MODE}" == "joint_tm" ]]; then
  guidance_mode_suffix="jointtm"
fi

track_points_suffix="pall"
if [[ "${TRACK_MAX_POINTS}" != "-1" ]]; then
  track_points_suffix="p${TRACK_MAX_POINTS}"
fi

stride_suffix="auto"
if [[ "${WAN_MOVE_TEMPORAL_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  stride_suffix="${WAN_MOVE_TEMPORAL_STRIDE}"
fi

for ckpt in "${ckpt_list[@]}"; do
  TRANSFORMER_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}/checkpoint-${ckpt}"
  if [[ ! -d "${TRANSFORMER_CHECKPOINT_PATH}" ]]; then
    echo "[davis_eval_wan_move] checkpoint not found: ${TRANSFORMER_CHECKPOINT_PATH}"
    exit 1
  fi

  OUTPUT_DIR_BASE="${SAVE_BASE_DIR_BASE}/davis_track_eval/${EXP_NAME}/checkpoint-${ckpt}"
  OUTPUT_NAME_SUFFIX_RUN="${OUTPUT_NAME_SUFFIX}"
  if [[ -z "${OUTPUT_NAME_SUFFIX_RUN}" ]]; then
    OUTPUT_NAME_SUFFIX_RUN="${guidance_mode_suffix}_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_${track_points_suffix}_wanmove_stride${stride_suffix}_seed${SEED}"
  fi
  OUTPUT_DIR="${OUTPUT_DIR_BASE}/${OUTPUT_NAME_SUFFIX_RUN}"
  mkdir -p "${OUTPUT_DIR}"

  echo "[davis_eval_wan_move] checkpoint=${TRANSFORMER_CHECKPOINT_PATH}"
  echo "[davis_eval_wan_move] output_name_suffix=${OUTPUT_NAME_SUFFIX_RUN}"
  echo "[davis_eval_wan_move] output_dir=${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  "${PYTHON_BIN}" evaluation/davis_track_eval.py \
    --davis_root "${DAVIS_ROOT}" \
    --model_name "${MODEL_NAME}" \
    --config_path "${CONFIG_PATH}" \
    --transformer_checkpoint_path "${TRANSFORMER_CHECKPOINT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_videos "${MAX_VIDEOS}" \
    --grid_size 50 \
    --track_max_points "${TRACK_MAX_POINTS}" \
    --video_length "${VIDEO_LENGTH}" \
    --reuse_generated_frames "${REUSE_GENERATED_FRAMES}" \
    --num_inference_steps "${NUM_STEPS}" \
    --seed "${SEED}" \
    --guidance_mode "${GUIDANCE_MODE}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --motion_guidance_weight "${MOTION_GUIDANCE_WEIGHT}" \
    --text_guidance_weight "${TEXT_GUIDANCE_WEIGHT}" \
    --normalize_track "${TRACK_NORMALIZE}" \
    --track_normalize_height "${TRACK_NORMALIZE_HEIGHT}" \
    --track_normalize_width "${TRACK_NORMALIZE_WIDTH}" \
    --track_condition_mode "${TRACK_CONDITION_MODE}" \
    --wan_move_temporal_stride "${WAN_MOVE_TEMPORAL_STRIDE}" \
    --track_point_sample_mode "${TRACK_POINT_SAMPLE_MODE}" \
    --track_sort_selected_indices "${TRACK_SORT_SELECTED_INDICES}" \
    --track_point_id_mode "${TRACK_POINT_ID_MODE}" \
    --track_point_sample_seed "${TRACK_POINT_SAMPLE_SEED}" \
    --track_overlay_trace_frames "${TRACK_OVERLAY_TRACE_FRAMES}" \
    --metrics "${METRIC_ARGS[@]}" \
    --lpips_net "${LPIPS_NET}" \
    --lpips_batch_size "${LPIPS_BATCH_SIZE}" \
    --lpips_device "${LPIPS_DEVICE}" \
    --track_overlay_scale "${TRACK_OVERLAY_SCALE}" \
    --track_overlay_crf "${TRACK_OVERLAY_CRF}" \
    --gt_track_cache_dir "${GT_TRACK_CACHE_DIR}" \
    --overwrite_gt_track_cache "${OVERWRITE_GT_TRACK_CACHE}" \
    --negative_text_feature_path "${NEGATIVE_TEXT_FEATURE_PATH}"
done
