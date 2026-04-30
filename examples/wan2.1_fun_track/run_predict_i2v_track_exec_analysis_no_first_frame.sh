#!/usr/bin/env bash
set -euo pipefail

# Metadata-based analysis loop for no-first-frame / no-track ablation.
VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json"
SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9)

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_lr_new_block_1e-4_early_blocks_5e-6_NEW_PARAMS_ONLY_STEPS_4400/checkpoint-4400"
TRANSFORMER_CHECKPOINT_PATH=""
SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-no-first-frame"

GUIDANCE_MODE="${GUIDANCE_MODE:-cfg}"

for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
    SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${SAMPLE_INDEX}"
    mkdir -p "${SAVE_DIR}"

    OUTPUT_NAME_SUFFIX="noff_notrack_idx_${SAMPLE_INDEX}"
    CUDA_VISIBLE_DEVICES=1 \
    TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
    METADATA_PATH="${VAL_METADATA_PATH}" \
    SAMPLE_INDEX="${SAMPLE_INDEX}" \
    USE_PROMPT_FROM_METADATA=true \
    GUIDANCE_MODE="${GUIDANCE_MODE}" \
    SAVE_DIR="${SAVE_DIR}" \
    OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX}" \
    bash examples/wan2.1_fun_track/run_predict_i2v_track_no_first_frame.sh
done
