#!/usr/bin/env bash
set -euo pipefail

# Metadata-based analysis loop (motion-only guidance with text null).
PYTHON_BIN="${PYTHON_BIN:-python}"
# VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json"
VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_train.json"
VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_train_selected64.json"
# VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_bin8_train_78k.json"
TRAIN_DATA_DIR="/data/shared-vilab/datasets/OpenVid-1M"
# SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
# SAMPLE_INDEX_LIST=(29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0)
# SAMPLE_INDEX_LIST=(27 26 19 7 2)
# SAMPLE_INDEX_LIST=(27)
# SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
# SAMPLE_INDEX_LIST=(19 7 2)
SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10)
TRACK_LATENT_SCALE_LIST=(1.0)
# Optional split scale. Leave empty to use TRACK_LATENT_SCALE for both.
# Example: TRACK_LATENT_FIRST_FRAME_SCALE=1.0, TRACK_LATENT_REST_FRAME_SCALE=4.0
TRACK_LATENT_FIRST_FRAME_SCALE="${TRACK_LATENT_FIRST_FRAME_SCALE:-0.5}"
TRACK_LATENT_REST_FRAME_SCALE="${TRACK_LATENT_REST_FRAME_SCALE:-2.0}"
SEED_LIST=(42 41)
# SEED_LIST=(39 38 37 36 35 34 33 32)
# SEED_LIST=(42 41 40)
# SEED_LIST=(42)

ckpt_list=(1700 1600 1500 1400 1300 1200 1100 1000 900)
ckpt_list=(1200 1100 1000 900)
ckpt_list=(2300)
ckpt_list=(1700)
ckpt_list=(1800 2300 2800)
ckpt_list=(2800 2300 1800)
ckpt_list=(4200 3800 3400 3000 2600 2200 1800 1400 1000 600 200)
ckpt_list=(5600)
ckpt_list=(200 400 600 800 1000 1200 1400 1600 1800)
ckpt_list=(1400 1200 1000 800 600 400 200)
ckpt_list=(1600)
ckpt_list=(9400)
ckpt_list=(1000)
# /data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-2800
# EXP_NAME="wan_track_track-patch-embed-init-track-alpha_1p0_init_noise_0p01_selected64_h_dim_256_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_selected64_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track-alpha_1p0_init_noise_0p01_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_selected64_h_dim_256_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_256_dropout_first-frame_0p1_text_0p1_track_0p1"
EXP_NAME="wan_track_track-patch-embed-init-track_local-point-id-mode_bs64_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1"
# EXP_NAME="wan_track_track-patch-embed-init-track_local-point-id-mode_bs64_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_256_dropout_first-frame_0p1_text_0p1_track_0p1"
# EXP_NAME="wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1"
# EXP_NAME="wan_track_patch-copy_first_gain-1.0_scale-8.0_track_local-point-id_bs64_train_78k_h_dim_64_all-dropouts_0p1"
EXP_NAME="wan_track_patch-copy_first_gain-1.0_scale-4.0_track_local-point-id_bs64_train_78k_h_dim_64_all-dropouts_0p1"
CHECKPOINT_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints"
SAVE_BASE_DIR_BASE="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track"

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

# Track normalization visualization options.
VISUALIZE_TRACK_NORMALIZATION="${VISUALIZE_TRACK_NORMALIZATION:-true}"
VIS_TRACK_MODES="${VIS_TRACK_MODES:-auto normalized pixel}"
VIS_TRACK_SAVE_MP4="${VIS_TRACK_SAVE_MP4:-false}"
VIS_TRACK_MAX_FRAMES="${VIS_TRACK_MAX_FRAMES:-32}"
VIS_TRACK_FPS="${VIS_TRACK_FPS:-8}"
VIS_TRACK_HEAT_PERCENTILE="${VIS_TRACK_HEAT_PERCENTILE:-99.0}"
VIS_TRACK_HEAT_GAMMA="${VIS_TRACK_HEAT_GAMMA:-0.55}"

for ckpt in "${ckpt_list[@]}"; do
    TRANSFORMER_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}/checkpoint-${ckpt}"
    SAVE_BASE_DIR="${SAVE_BASE_DIR_BASE}/${EXP_NAME}/checkpoint-${ckpt}/analysis_from_val_metadata_motion_trainset_recon"
    for SEED in "${SEED_LIST[@]}"; do
        for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
            for TRACK_LATENT_SCALE in "${TRACK_LATENT_SCALE_LIST[@]}"; do
                TRACK_LATENT_FIRST_FRAME_SCALE_RUN="${TRACK_LATENT_FIRST_FRAME_SCALE}"
                TRACK_LATENT_REST_FRAME_SCALE_RUN="${TRACK_LATENT_REST_FRAME_SCALE}"
                TRACK_LATENT_SCALE_SUFFIX="s${TRACK_LATENT_SCALE}"
                SAVE_DIR_SCALE_SUFFIX="scale_${TRACK_LATENT_SCALE}"
                if [[ -n "${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}" || -n "${TRACK_LATENT_REST_FRAME_SCALE_RUN}" ]]; then
                    TRACK_LATENT_FIRST_FRAME_SCALE_RUN="${TRACK_LATENT_FIRST_FRAME_SCALE_RUN:-${TRACK_LATENT_SCALE}}"
                    TRACK_LATENT_REST_FRAME_SCALE_RUN="${TRACK_LATENT_REST_FRAME_SCALE_RUN:-${TRACK_LATENT_SCALE}}"
                    TRACK_LATENT_SCALE_SUFFIX="sf${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}_sr${TRACK_LATENT_REST_FRAME_SCALE_RUN}"
                    SAVE_DIR_SCALE_SUFFIX="scale_first_${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}_rest_${TRACK_LATENT_REST_FRAME_SCALE_RUN}"
                fi
                SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${SAMPLE_INDEX}/${SAVE_DIR_SCALE_SUFFIX}"
                mkdir -p "${SAVE_DIR}"
                DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
                TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"
                TRACK_POINTS_SUFFIX="pall"
                if [[ "${TRACK_MAX_POINTS}" != "-1" ]]; then
                    TRACK_POINTS_SUFFIX="p${TRACK_MAX_POINTS}"
                fi
                OUTPUT_NAME_SUFFIX="motiononly_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_${TRACK_POINTS_SUFFIX}_ff${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}_rf${TRACK_LATENT_REST_FRAME_SCALE_RUN}_toff${TRACK_CONDITION_INDEX_OFFSET}_seed${SEED}"
                CUDA_VISIBLE_DEVICES=5 \
                TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
                METADATA_PATH="${VAL_METADATA_PATH}" \
                TRAIN_DATA_DIR="${TRAIN_DATA_DIR}" \
                SAMPLE_INDEX="${SAMPLE_INDEX}" \
                USE_PROMPT_FROM_METADATA=true \
                COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
                SAVE_DIR="${SAVE_DIR}" \
                OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX}" \
                OVERLAY_PAD_VALUE=0 \
                OVERLAY_LINEWIDTH=1 \
                OVERLAY_TRACE_FRAMES=8 \
                TRACK_MAX_POINTS="${TRACK_MAX_POINTS}" \
                TRACK_NORMALIZE="${TRACK_NORMALIZE}" \
                TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM}" \
                TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE}" \
                TRACK_LATENT_FIRST_FRAME_SCALE="${TRACK_LATENT_FIRST_FRAME_SCALE_RUN}" \
                TRACK_LATENT_REST_FRAME_SCALE="${TRACK_LATENT_REST_FRAME_SCALE_RUN}" \
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
done



# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_lr_new_block_1e-4_early_blocks_5e-6_NEW_PARAMS_ONLY_STEPS_4400/checkpoint-7800"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_lr_new_block_1e-4_early_blocks_5e-6_NEW_PARAMS_ONLY_STEPS_4400/checkpoint-7800/analysis_from_val_metadata_motion_only"

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_textdrop1_lr_new_block_1e-4_all_blocks_5e-6_fork-icy-silence-27/checkpoint-3200"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_textdrop1_lr_new_block_1e-4_all_blocks_5e-6_fork-icy-silence-27/checkpoint-3200/analysis_from_val_metadata_joint"

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_normalize_fixed_Openvid-0p6M_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-1200"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_normalize_fixed_Openvid-0p6M_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-1200/analysis_from_val_metadata_motion_only"
# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track-alpha_0p5_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-7000"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_track-patch-embed-init-track-alpha_0p5_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-7000/analysis_from_val_metadata_motion_only_trainset"

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track-alpha_0p5_fork_deft-sunset-41-4000ckpt_dropout_0p1s_OpenVid-1M_550000/checkpoint-8800"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_track-patch-embed-init-track-alpha_0p5_fork_deft-sunset-41-4000ckpt_dropout_0p1s_OpenVid-1M_550000/checkpoint-8800/analysis_from_val_metadata_motion_only_trainset"

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track-alpha_1p0_init_noise_0p01_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-400"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_track-patch-embed-init-track-alpha_1p0_init_noise_0p01_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-400/analysis_from_val_metadata_motion_only_trainset"
