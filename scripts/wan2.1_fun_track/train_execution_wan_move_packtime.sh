#!/usr/bin/env bash
# Wan-Move-PackTime track conditioning:
# keep the Wan-Move inpaint-latent skip path, pack 4 pixel-time replicated
# track feature maps per Wan latent frame, and learn a zero-init residual adapter.
TRAIN_EXECUTION_SCRIPT_TRACK="${BASH_SOURCE[0]}" \
GRADIENT_CHECKPOINTING_TRACK=true \
MIXED_PRECISION_TRACK=bf16 \
NUM_PROCESSES_TRACK=8 \
DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M" \
DATASET_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track_fineMotion_bin8_train_78k.json" \
VAL_DATA_META_NAME_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json" \
CHECKPOINT_DIR_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_wan_move_packtime_condition_bin8_train_78k_dropout_first-frame_0p1_text_0p1_track_0p1" \
CHECKPOINTING_STEPS_TRACK=200 \
VALIDATION_STEPS_TRACK=200 \
VALIDATION_MAX_BATCHES_TRACK=4 \
TRACK_CONDITION_MODE_TRACK=wan_move_packtime \
WAN_MOVE_PACKTIME_HIDDEN_DIM_TRACK=64 \
TRACK_CONDITION_DROP_PROB_TRACK=0.1 \
FIRST_FRAME_CONDITION_DROP_PROB_TRACK=0.1 \
TEXT_DROP_RATIO_TRACK=0.1 \
DEBUG_WEIGHT_UPDATE_TRACK=false \
DEBUG_WEIGHT_UPDATE_TOPK_TRACK=50 \
TRACK_DEBUG_VIS_STEPS_TRACK=0 \
NUM_TRAIN_EPOCHS_TRACK=1000 \
APPLY_TRACK_PATCH_EMBED_INIT_TRACK=false \
TRAIN_BATCH_SIZE_TRACK=8 \
NEW_PARAMS_ONLY_STEPS_TRACK=0 \
LEARNING_RATE_TRACK=1e-5 \
ADAM_WEIGHT_DECAY_TRACK=0 \
ADAM_EPSILON_TRACK=1e-8 \
TRACK_RANDOM_POINTS_MIN_TRACK=1 \
TRACK_RANDOM_POINTS_MAX_TRACK=200 \
NEW_TRACK_LAYERS_LR_TRACK=1e-5 \
EARLY_BLOCKS_LR_TRACK=1e-5 \
TRAIN_EARLY_BLOCKS_TRACK=-1 \
LR_WARMUP_STEPS_TRACK=10 \
PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK="/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz" \
bash scripts/wan2.1_fun_track/train_track.sh
