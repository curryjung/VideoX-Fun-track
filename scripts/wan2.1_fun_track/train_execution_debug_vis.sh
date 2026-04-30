#!/usr/bin/env bash
# Dense debug run for track visualization.
# - Saves track debug videos every optimizer step (TRACK_DEBUG_VIS_STEPS_TRACK=1).
# - Keeps full frame length (TRACK_DEBUG_VIS_MAX_FRAMES_TRACK=-1) to avoid truncation.
# - Outputs are written under:
#   /data/project-vilab/jaeseok/VideoX-Fun/output_dir_wan2.1_fun_track_debug_vis_dense/track_debug_vis_dense/step_*
GRADIENT_CHECKPOINTING_TRACK=true \
MIXED_PRECISION_TRACK=bf16 \
NUM_PROCESSES_TRACK=4 \
DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M" \
DATASET_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_train.json" \
VAL_DATA_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json" \
OUTPUT_DIR_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/output_dir_wan2.1_fun_track_debug_vis_dense" \
CHECKPOINT_DIR_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_debug_vis_dense" \
CHECKPOINTING_STEPS_TRACK=200 \
VALIDATION_STEPS_TRACK=0 \
VALIDATION_MAX_BATCHES_TRACK=4 \
TRACK_CONDITION_DROP_PROB_TRACK=0.1 \
FIRST_FRAME_CONDITION_DROP_PROB_TRACK=0.1 \
TEXT_DROP_RATIO_TRACK=0.1 \
DEBUG_WEIGHT_UPDATE_TRACK=false \
DEBUG_WEIGHT_UPDATE_TOPK_TRACK=50 \
TRACK_DEBUG_VIS_STEPS_TRACK=1 \
TRACK_DEBUG_VIS_DIR_TRACK="track_debug_vis_dense" \
TRACK_DEBUG_VIS_SAMPLE_INDEX_TRACK=0 \
TRACK_DEBUG_VIS_MAX_FRAMES_TRACK=-1 \
TRACK_DEBUG_VIS_MAX_POINTS_TRACK=1024 \
TRACK_DEBUG_VIS_FPS_TRACK=16 \
MAX_TRAIN_STEPS_TRACK=24 \
NEW_PARAMS_ONLY_STEPS_TRACK=0 \
LEARNING_RATE_TRACK=1e-5 \
NEW_TRACK_LAYERS_LR_TRACK=1e-5 \
EARLY_BLOCKS_LR_TRACK=1e-5 \
TRAIN_EARLY_BLOCKS_TRACK=-1 \
LR_WARMUP_STEPS_TRACK=10 \
TRACK_SORT_SELECTED_INDICES_TRACK=false \
PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz" \
bash scripts/wan2.1_fun_track/train_track.sh
