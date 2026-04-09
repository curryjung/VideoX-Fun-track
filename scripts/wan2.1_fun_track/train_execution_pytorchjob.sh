#!/usr/bin/env bash
# Environment-only launcher for PyTorchJob runs.
# Adjust env values here or override them from Job YAML env section.

GRADIENT_CHECKPOINTING_TRACK=true \
MIXED_PRECISION_TRACK=bf16 \
NUM_PROCESSES_TRACK="${NUM_PROCESSES_TRACK:-${PET_NPROC_PER_NODE:-8}}" \
NUM_MACHINES_TRACK="${NUM_MACHINES_TRACK:-${PET_NNODES:-2}}" \
RDZV_ID_TRACK="${RDZV_ID_TRACK:-${PET_RDZV_ID:-wan2-track}}" \
RDZV_BACKEND_TRACK="${RDZV_BACKEND_TRACK:-${PET_RDZV_BACKEND:-c10d}}" \
RDZV_ENDPOINT_TRACK="${RDZV_ENDPOINT_TRACK:-${PET_RDZV_ENDPOINT:-127.0.0.1:29500}}" \
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
NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" \
GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-AF_INET}" \
bash scripts/wan2.1_fun_track/train_track_pytorchjob.sh
