#!/usr/bin/env bash
set -euo pipefail

source .venv-videoxfun/bin/activate

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-4}"
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-200}"
MAX_VIDEOS="${MAX_VIDEOS:-20}"
VIDEO_LENGTH="${VIDEO_LENGTH:-81}"
NUM_STEPS="${NUM_STEPS:-50}"
SEED="${SEED:-42}"

#Experiment settings
TRACK_LATENT_FIRST_FRAME_SCALE="${TRACK_LATENT_FIRST_FRAME_SCALE:-0.5}"
TRACK_LATENT_REST_FRAME_SCALE="${TRACK_LATENT_REST_FRAME_SCALE:-1.8}"
GUIDANCE_MODE="${GUIDANCE_MODE:-motion_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-1.5}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.0}"

TRACK_LATENT_SCALE_LIST=(1.0)
# ckpt_list=(15200 13000 11000 9000 7000 5000 3000 1000)
ckpt_list=(17200)
# ckpt_list=(12200)
# ckpt_list=(20000 18000 16000 14000 12000 10200)
#Checkpoint path
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/data/project-vilab/jaeseok/VideoX-Fun/checkpoints}"

# save base directory
SAVE_BASE_DIR_BASE="${SAVE_BASE_DIR_BASE:-/data/project-vilab/jaeseok/VideoX-Fun/evaluation/results}"
GT_TRACK_CACHE_DIR="${GT_TRACK_CACHE_DIR:-${SAVE_BASE_DIR_BASE}/davis_gt_track_cache}"
OVERWRITE_GT_TRACK_CACHE="${OVERWRITE_GT_TRACK_CACHE:-false}"
EXP_NAME="${EXP_NAME:-wan_track_init-proud-sea-57-ckpt1000_ff-scale-0.5_rf-scale-1.8_openvid-0p6m_wisa-80k}"
# EXP_NAME="${EXP_NAME:-wan_track_patch-copy_first_gain-1.0_scale-4.0_track_local-point-id_bs64_train_78k_h_dim_64_all-dropouts_0p1}"
# default settings
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-1000}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-random}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-false}"
TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE:-local}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-42}"
WAN_MOVE_TEMPORAL_STRIDE="${WAN_MOVE_TEMPORAL_STRIDE:-1}"
TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM:-64}"
TRACK_CONDITION_MODE="${TRACK_CONDITION_MODE:-track_head}"
TRACK_OVERLAY_TRACE_FRAMES="${TRACK_OVERLAY_TRACE_FRAMES:-8}"
TRACK_OVERLAY_SCALE="${TRACK_OVERLAY_SCALE:-0.5}"
TRACK_OVERLAY_CRF="${TRACK_OVERLAY_CRF:-16}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-}"

# evaluation metrics / artifact reuse
METRICS="${METRICS:-epe psnr ssim lpips}"
REUSE_GENERATED_FRAMES="${REUSE_GENERATED_FRAMES:-never}"
LPIPS_NET="${LPIPS_NET:-alex}"
LPIPS_BATCH_SIZE="${LPIPS_BATCH_SIZE:-16}"
LPIPS_DEVICE="${LPIPS_DEVICE:-}"

read -r -a METRIC_ARGS <<< "${METRICS}"



for ckpt in "${ckpt_list[@]}"; do
    TRANSFORMER_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}/checkpoint-${ckpt}"
    OUTPUT_DIR_BASE="${SAVE_BASE_DIR_BASE}/davis_track_eval/${EXP_NAME}/checkpoint-${ckpt}"
    for TRACK_LATENT_SCALE in "${TRACK_LATENT_SCALE_LIST[@]}"; do
        TRACK_LATENT_FIRST_FRAME_SCALE_RUN="${TRACK_LATENT_FIRST_FRAME_SCALE}"
        TRACK_LATENT_REST_FRAME_SCALE_RUN="${TRACK_LATENT_REST_FRAME_SCALE}"
        if [[ -n "${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}" || -n "${TRACK_LATENT_REST_FRAME_SCALE_RUN}" ]]; then
            TRACK_LATENT_FIRST_FRAME_SCALE_RUN="${TRACK_LATENT_FIRST_FRAME_SCALE_RUN:-${TRACK_LATENT_SCALE}}"
            TRACK_LATENT_REST_FRAME_SCALE_RUN="${TRACK_LATENT_REST_FRAME_SCALE_RUN:-${TRACK_LATENT_SCALE}}"
        fi

        TRACK_POINTS_SUFFIX="pall"
        if [[ "${TRACK_MAX_POINTS}" != "-1" ]]; then
            TRACK_POINTS_SUFFIX="p${TRACK_MAX_POINTS}"
        fi
        TRACK_CONDITION_MODE_SUFFIX=""
        if [[ "${TRACK_CONDITION_MODE}" != "track_head" ]]; then
            TRACK_CONDITION_MODE_SUFFIX="_${TRACK_CONDITION_MODE}"
        fi
        GUIDANCE_MODE_SUFFIX="${GUIDANCE_MODE}"
        if [[ "${GUIDANCE_MODE}" == "motion_only" ]]; then
            GUIDANCE_MODE_SUFFIX="motiononly"
        elif [[ "${GUIDANCE_MODE}" == "text_only" ]]; then
            GUIDANCE_MODE_SUFFIX="textonly"
        elif [[ "${GUIDANCE_MODE}" == "joint_tm" ]]; then
            GUIDANCE_MODE_SUFFIX="jointtm"
        fi

        OUTPUT_NAME_SUFFIX_RUN="${OUTPUT_NAME_SUFFIX}"
        if [[ -z "${OUTPUT_NAME_SUFFIX_RUN}" ]]; then
            OUTPUT_NAME_SUFFIX_RUN="${GUIDANCE_MODE_SUFFIX}_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_${TRACK_POINTS_SUFFIX}_ff${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}_rf${TRACK_LATENT_REST_FRAME_SCALE_RUN}${TRACK_CONDITION_MODE_SUFFIX}_seed${SEED}"
        fi
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${OUTPUT_NAME_SUFFIX_RUN}"
        mkdir -p "${OUTPUT_DIR}"

        echo "[davis_eval] output_name_suffix=${OUTPUT_NAME_SUFFIX_RUN}"
        echo "[davis_eval] output_dir=${OUTPUT_DIR}"

        #Run evaluation
        CUDA_VISIBLE_DEVICES="${GPU}" \
        TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        "${PYTHON_BIN}" evaluation/davis_track_eval.py \
          --davis_root /data/project-vilab/jaeseok/davis/DAVIS/JPEGImages/480p \
          --model_name models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP \
          --config_path config/wan2.1/wan_civitai.yaml \
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
          --motion_guidance_weight "${MOTION_GUIDANCE_WEIGHT}" \
          --text_guidance_weight "${TEXT_GUIDANCE_WEIGHT}" \
          --track_head_hidden_dim "${TRACK_HEAD_HIDDEN_DIM}" \
          --track_condition_mode "${TRACK_CONDITION_MODE}" \
          --track_point_sample_mode "${TRACK_POINT_SAMPLE_MODE}" \
          --track_sort_selected_indices "${TRACK_SORT_SELECTED_INDICES}" \
          --track_point_id_mode "${TRACK_POINT_ID_MODE}" \
          --track_point_sample_seed "${TRACK_POINT_SAMPLE_SEED}" \
          --wan_move_temporal_stride "${WAN_MOVE_TEMPORAL_STRIDE}" \
          --track_overlay_trace_frames "${TRACK_OVERLAY_TRACE_FRAMES}" \
          --metrics "${METRIC_ARGS[@]}" \
          --lpips_net "${LPIPS_NET}" \
          --lpips_batch_size "${LPIPS_BATCH_SIZE}" \
          --lpips_device "${LPIPS_DEVICE}" \
          --track_latent_scale "${TRACK_LATENT_SCALE}" \
          --track_latent_first_frame_scale "${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}" \
          --track_latent_rest_frame_scale "${TRACK_LATENT_REST_FRAME_SCALE_RUN}" \
          --track_overlay_scale "${TRACK_OVERLAY_SCALE}" \
          --track_overlay_crf "${TRACK_OVERLAY_CRF}" \
          --gt_track_cache_dir "${GT_TRACK_CACHE_DIR}" \
          --overwrite_gt_track_cache "${OVERWRITE_GT_TRACK_CACHE}" \
          --negative_text_feature_path /data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz
    done
done
