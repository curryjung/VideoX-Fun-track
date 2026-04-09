#!/usr/bin/env bash

# Example for 2 nodes x 8 GPUs.
# Run this file on every node.
# On rank 0 node:
#   MACHINE_RANK_TRACK=0 MAIN_PROCESS_IP_TRACK=<rank0-ip> bash scripts/wan2.1_fun_track/train_execution_multinode.sh
# On rank 1 node:
#   MACHINE_RANK_TRACK=1 MAIN_PROCESS_IP_TRACK=<rank0-ip> bash scripts/wan2.1_fun_track/train_execution_multinode.sh

GRADIENT_CHECKPOINTING_TRACK=true \
MIXED_PRECISION_TRACK=bf16 \
NUM_PROCESSES_TRACK=16 \
NUM_MACHINES_TRACK=2 \
MACHINE_RANK_TRACK="${MACHINE_RANK_TRACK:-0}" \
MAIN_PROCESS_IP_TRACK="${MAIN_PROCESS_IP_TRACK:-127.0.0.1}" \
MAIN_PROCESS_PORT_TRACK="${MAIN_PROCESS_PORT_TRACK:-29500}" \
DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M" \
DATASET_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_train.json" \
VAL_DATA_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json" \
CHECKPOINT_DIR_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/init_w_NEW_PARAMS_ONLY_STEPS_600_lr_1e-5" \
CHECKPOINTING_STEPS_TRACK=200 \
VALIDATION_STEPS_TRACK=200 \
VALIDATION_MAX_BATCHES_TRACK=4 \
TRACK_CONDITION_DROP_PROB_TRACK=0.1 \
DEBUG_WEIGHT_UPDATE_TRACK=false \
DEBUG_WEIGHT_UPDATE_TOPK_TRACK=50 \
INIT_MODEL_FROM_CHECKPOINT_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_NEW_PARAMS_ONLY_STEPS_1000/checkpoint-600" \
bash scripts/wan2.1_fun_track/train_track_multinode.sh
