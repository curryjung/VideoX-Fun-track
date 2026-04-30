#!/usr/bin/env bash
set -euo pipefail

# Metadata-based analysis loop
VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json"
TRAIN_DATA_DIR="/data/shared-vilab/datasets/OpenVid-1M"
SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9)
TRACK_LATENT_SCALE_LIST=(1.0)

TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_lr_new_block_1e-4_early_blocks_5e-6_NEW_PARAMS_ONLY_STEPS_4400/checkpoint-4400"
SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_lr_new_block_1e-4_early_blocks_5e-6_NEW_PARAMS_ONLY_STEPS_4400/checkpoint-4400/analysis_from_val_metadata"

for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
    for TRACK_LATENT_SCALE in "${TRACK_LATENT_SCALE_LIST[@]}"; do
        SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${SAMPLE_INDEX}/scale_${TRACK_LATENT_SCALE}"
        mkdir -p "${SAVE_DIR}"
        DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
        TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"
        OUTPUT_NAME_SUFFIX="track_on_p2000_s${TRACK_LATENT_SCALE}"
        CUDA_VISIBLE_DEVICES=1 \
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
        TRACK_MAX_POINTS=2000 \
        TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE}" \
        DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
        TRACK_ANALYSIS="${TRACK_ANALYSIS}" \
        bash examples/wan2.1_fun_track/run_predict_i2v_track.sh
    done
done
