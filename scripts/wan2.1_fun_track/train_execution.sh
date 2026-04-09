#!/usr/bin/env bash
GRADIENT_CHECKPOINTING_TRACK=true \
MIXED_PRECISION_TRACK=bf16 \
NUM_PROCESSES_TRACK=8 \
DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M" \
DATASET_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_train.json" \
VAL_DATA_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json" \
CHECKPOINT_DIR_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_NEW_PARAMS_ONLY_STEPS_10000" \
CHECKPOINTING_STEPS_TRACK=200 \
VALIDATION_STEPS_TRACK=200 \
VALIDATION_MAX_BATCHES_TRACK=4 \
TRACK_CONDITION_DROP_PROB_TRACK=0.1 \
DEBUG_WEIGHT_UPDATE_TRACK=false \
DEBUG_WEIGHT_UPDATE_TOPK_TRACK=50 \
RESUME_FROM_CHECKPOINT_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_NEW_PARAMS_ONLY_STEPS_10000/checkpoint-2000" \
NEW_PARAMS_ONLY_STEPS_TRACK=10000 \
LEARNING_RATE_TRACK=5e-4 \
LR_WARMUP_STEPS_TRACK=800 \
bash scripts/wan2.1_fun_track/train_track.sh



# RESUME_FROM_CHECKPOINT_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/output_dir_wan2.1_fun_track/checkpoint-1750" \
# DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M/out_preprocess_openvid_cotracker_preshard_20260402_134900" \
