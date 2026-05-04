#!/usr/bin/env bash
# Prerequisites (once per venv / environment):
#   cd /path/to/VideoX-Fun && pip install -e .
# This script assumes `videox_fun` is importable from that install (no PYTHONPATH hacks).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHON_BIN_TRACK="${PYTHON_BIN_TRACK:-python}"
export NUM_PROCESSES_TRACK="${NUM_PROCESSES_TRACK:-1}"
export MIXED_PRECISION_TRACK="${MIXED_PRECISION_TRACK:-bf16}"
export GRADIENT_CHECKPOINTING_TRACK="${GRADIENT_CHECKPOINTING_TRACK:-false}"

if ! "${PYTHON_BIN_TRACK}" -c "import videox_fun" 2>/dev/null; then
  echo "[error] Cannot import videox_fun. Install VideoX-Fun in editable mode from the repo root:"
  echo "  cd \"${REPO_ROOT}\" && ${PYTHON_BIN_TRACK} -m pip install -e ."
  exit 1
fi

export MODEL_NAME_TRACK="${MODEL_NAME_TRACK:-models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP}"
export DATASET_NAME_TRACK="${DATASET_NAME_TRACK:-datasets/internal_datasets/}"
export DATASET_META_NAME_TRACK="${DATASET_META_NAME_TRACK:-datasets/internal_datasets/metadata_track.json}"
export DATASET_SPECS_TRACK="${DATASET_SPECS_TRACK:-}"
export INPUT_MODE_TRACK="${INPUT_MODE_TRACK:-latent}"
export DATASET_ROOT_MAP_JSON_TRACK="${DATASET_ROOT_MAP_JSON_TRACK:-}"
export DATASET_ROOT_ID_KEY_TRACK="${DATASET_ROOT_ID_KEY_TRACK:-root_id}"
export TRACK_MAX_POINTS_TRACK="${TRACK_MAX_POINTS_TRACK:--1}"
export TRACK_RANDOM_POINTS_MIN_TRACK="${TRACK_RANDOM_POINTS_MIN_TRACK:-1000}"
export TRACK_RANDOM_POINTS_MAX_TRACK="${TRACK_RANDOM_POINTS_MAX_TRACK:-2500}"
export TRACK_SORT_SELECTED_INDICES_TRACK="${TRACK_SORT_SELECTED_INDICES_TRACK:-true}"
export TRACK_POINT_ID_MODE_TRACK="${TRACK_POINT_ID_MODE_TRACK:-original}"
export TRACK_CONDITION_DROP_PROB_TRACK="${TRACK_CONDITION_DROP_PROB_TRACK:-0.0}"
export TRACK_CONDITION_MODE_TRACK="${TRACK_CONDITION_MODE_TRACK:-track_head}"
export APPLY_TRACK_PATCH_EMBED_INIT_TRACK="${APPLY_TRACK_PATCH_EMBED_INIT_TRACK:-true}"
export TRACK_PATCH_INIT_MODE_TRACK="${TRACK_PATCH_INIT_MODE_TRACK:-copy_noisy}"
export TRACK_PATCH_INIT_GAIN_TRACK="${TRACK_PATCH_INIT_GAIN_TRACK:-1.0}"
export TRACK_LATENT_SCALE_TRACK="${TRACK_LATENT_SCALE_TRACK:-1.0}"
export TRACK_LATENT_FIRST_FRAME_SCALE_TRACK="${TRACK_LATENT_FIRST_FRAME_SCALE_TRACK:-${TRACK_LATENT_FIRST_FRAME_SCALE:-}}"
export TRACK_LATENT_REST_FRAME_SCALE_TRACK="${TRACK_LATENT_REST_FRAME_SCALE_TRACK:-${TRACK_LATENT_REST_FRAME_SCALE:-}}"
export ADD_TRACK_INIT_NOISE_TRACK="${ADD_TRACK_INIT_NOISE_TRACK:-false}"
export TRACK_INIT_NOISE_SCALE_TRACK="${TRACK_INIT_NOISE_SCALE_TRACK:-0.01}"
export TRACK_HEAD_HIDDEN_DIM_TRACK="${TRACK_HEAD_HIDDEN_DIM_TRACK:-}"
export FIRST_FRAME_CONDITION_DROP_PROB_TRACK="${FIRST_FRAME_CONDITION_DROP_PROB_TRACK:-0.0}"
export VAL_DATA_META_NAME_TRACK="${VAL_DATA_META_NAME_TRACK:-}"
export VAL_DATASET_NAME_TRACK="${VAL_DATASET_NAME_TRACK:-}"
export VALIDATION_STEPS_TRACK="${VALIDATION_STEPS_TRACK:-0}"
export VALIDATION_MAX_BATCHES_TRACK="${VALIDATION_MAX_BATCHES_TRACK:-8}"
export NEW_PARAMS_ONLY_STEPS_TRACK="${NEW_PARAMS_ONLY_STEPS_TRACK:-0}"
export DEBUG_WEIGHT_UPDATE_TRACK="${DEBUG_WEIGHT_UPDATE_TRACK:-false}"
export DEBUG_WEIGHT_UPDATE_TOPK_TRACK="${DEBUG_WEIGHT_UPDATE_TOPK_TRACK:-30}"
export TRACK_DEBUG_VIS_STEPS_TRACK="${TRACK_DEBUG_VIS_STEPS_TRACK:-0}"
export TRACK_DEBUG_VIS_DIR_TRACK="${TRACK_DEBUG_VIS_DIR_TRACK:-track_debug_vis}"
export TRACK_DEBUG_VIS_SAMPLE_INDEX_TRACK="${TRACK_DEBUG_VIS_SAMPLE_INDEX_TRACK:-0}"
export TRACK_DEBUG_VIS_MAX_FRAMES_TRACK="${TRACK_DEBUG_VIS_MAX_FRAMES_TRACK:-32}"
export TRACK_DEBUG_VIS_MAX_POINTS_TRACK="${TRACK_DEBUG_VIS_MAX_POINTS_TRACK:-512}"
export TRACK_DEBUG_VIS_FPS_TRACK="${TRACK_DEBUG_VIS_FPS_TRACK:-8}"
export NUM_TRAIN_EPOCHS_TRACK="${NUM_TRAIN_EPOCHS_TRACK:-100}"
export MAX_TRAIN_STEPS_TRACK="${MAX_TRAIN_STEPS_TRACK:-}"
export TRAIN_BATCH_SIZE_TRACK="${TRAIN_BATCH_SIZE_TRACK:-8}"
export CHECKPOINTING_STEPS_TRACK="${CHECKPOINTING_STEPS_TRACK:-500}"
export LEARNING_RATE_TRACK="${LEARNING_RATE_TRACK:-1e-5}"
export ADAM_WEIGHT_DECAY_TRACK="${ADAM_WEIGHT_DECAY_TRACK:-3e-2}"
export ADAM_EPSILON_TRACK="${ADAM_EPSILON_TRACK:-1e-10}"
export NEW_TRACK_LAYERS_LR_TRACK="${NEW_TRACK_LAYERS_LR_TRACK:-${LEARNING_RATE_TRACK}}"
export EARLY_BLOCKS_LR_TRACK="${EARLY_BLOCKS_LR_TRACK:-${LEARNING_RATE_TRACK}}"
export TRAIN_EARLY_BLOCKS_TRACK="${TRAIN_EARLY_BLOCKS_TRACK:--1}"
export LR_WARMUP_STEPS_TRACK="${LR_WARMUP_STEPS_TRACK:-800}"
export TEXT_DROP_RATIO_TRACK="${TEXT_DROP_RATIO_TRACK:-0.1}"
export USE_FIRST_FRAME_CONDITION_TRACK="${USE_FIRST_FRAME_CONDITION_TRACK:-true}"

# Logging: "wandb" | "tensorboard" | "comet_ml" | "all" (wandb requires `pip install wandb` and auth).
export REPORT_TO_TRACK="${REPORT_TO_TRACK:-wandb}"
export WANDB_RUN_NAME_TRACK="${WANDB_RUN_NAME_TRACK:-}"

# Run directory (logs / trackers) and optional separate checkpoint root.
export OUTPUT_DIR_TRACK="${OUTPUT_DIR_TRACK:-output_dir_wan2.1_fun_track}"
export CHECKPOINT_DIR_TRACK="${CHECKPOINT_DIR_TRACK:-}"

# Resume: absolute path to checkpoint-* folder, or "latest" under CHECKPOINT_DIR_TRACK / OUTPUT_DIR_TRACK.
# Example: RESUME_FROM_CHECKPOINT_TRACK=/data/.../VideoX-Fun/output_dir_wan2.1_fun_track/checkpoint-1750
export RESUME_FROM_CHECKPOINT_TRACK="${RESUME_FROM_CHECKPOINT_TRACK:-}"
# Model-only init: load model weights from checkpoint-* but reset optimizer/lr scheduler/global_step.
# Example: INIT_MODEL_FROM_CHECKPOINT_TRACK=/data/.../checkpoints/.../checkpoint-600
export INIT_MODEL_FROM_CHECKPOINT_TRACK="${INIT_MODEL_FROM_CHECKPOINT_TRACK:-}"
# Optional: npz from scripts/wan2.1_fun_track/precompute_uncond_text_track.py (caption-drop rows in latent mode).
export PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK="${PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK:-}"

if [[ -n "${DATASET_SPECS_TRACK}" && ! -f "${DATASET_SPECS_TRACK}" ]]; then
  echo "[error] Dataset specs file not found: ${DATASET_SPECS_TRACK}"
  exit 1
fi

if [[ -z "${DATASET_SPECS_TRACK}" && ! -f "${DATASET_META_NAME_TRACK}" ]]; then
  echo "[error] Metadata file not found: ${DATASET_META_NAME_TRACK}"
  echo "[hint] Build it first with:"
  echo "  python scripts/wan2.1_fun_track/build_metadata_track.py --help"
  exit 1
fi

EXTRA_ARGS_TRACK=()
if [[ -n "${DATASET_ROOT_MAP_JSON_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--train_data_root_map_json_track=${DATASET_ROOT_MAP_JSON_TRACK}")
fi
if [[ -n "${DATASET_SPECS_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--train_dataset_specs_track=${DATASET_SPECS_TRACK}")
fi
if [[ -n "${DATASET_ROOT_MAP_JSON_TRACK}" || -n "${DATASET_SPECS_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--train_data_root_id_key_track=${DATASET_ROOT_ID_KEY_TRACK}")
fi
if [[ -n "${WANDB_RUN_NAME_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--wandb_run_name_track=${WANDB_RUN_NAME_TRACK}")
fi
if [[ -n "${CHECKPOINT_DIR_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--checkpoint_dir_track=${CHECKPOINT_DIR_TRACK}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--resume_from_checkpoint_track=${RESUME_FROM_CHECKPOINT_TRACK}")
fi
if [[ -n "${INIT_MODEL_FROM_CHECKPOINT_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--init_model_from_checkpoint_track=${INIT_MODEL_FROM_CHECKPOINT_TRACK}")
fi
if [[ -n "${TRACK_LATENT_FIRST_FRAME_SCALE_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--track_latent_first_frame_scale=${TRACK_LATENT_FIRST_FRAME_SCALE_TRACK}")
fi
if [[ -n "${TRACK_LATENT_REST_FRAME_SCALE_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--track_latent_rest_frame_scale=${TRACK_LATENT_REST_FRAME_SCALE_TRACK}")
fi
if [[ -n "${VAL_DATA_META_NAME_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--val_data_meta_track=${VAL_DATA_META_NAME_TRACK}")
fi
if [[ -n "${VAL_DATASET_NAME_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--val_data_dir_track=${VAL_DATASET_NAME_TRACK}")
fi
if [[ "${USE_FIRST_FRAME_CONDITION_TRACK}" == "true" ]]; then
  EXTRA_ARGS_TRACK+=("--use_first_frame_condition_track")
fi
if [[ "${DEBUG_WEIGHT_UPDATE_TRACK}" == "true" ]]; then
  EXTRA_ARGS_TRACK+=("--debug_weight_update_track" "--debug_weight_update_topk_track=${DEBUG_WEIGHT_UPDATE_TOPK_TRACK}")
fi
if [[ -n "${MAX_TRAIN_STEPS_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--max_train_steps=${MAX_TRAIN_STEPS_TRACK}")
fi
if [[ -n "${PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--precomputed_uncond_text_npz_track=${PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK}")
fi
if [[ -n "${TRACK_HEAD_HIDDEN_DIM_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--track_head_hidden_dim=${TRACK_HEAD_HIDDEN_DIM_TRACK}")
fi
if [[ -n "${TRAIN_BATCH_SIZE_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--train_batch_size=${TRAIN_BATCH_SIZE_TRACK}")
fi

# Save launcher snapshot for reproducibility at training start.
CHECKPOINT_ROOT_TRACK="${CHECKPOINT_DIR_TRACK:-${OUTPUT_DIR_TRACK}}"
RUN_META_DIR_TRACK="${CHECKPOINT_ROOT_TRACK}/run_meta"
mkdir -p "${RUN_META_DIR_TRACK}"
RUN_STAMP_TRACK="$(date +%Y%m%d_%H%M%S)"
cp "${BASH_SOURCE[0]}" "${RUN_META_DIR_TRACK}/train_track_${RUN_STAMP_TRACK}.sh"
TRAIN_EXECUTION_SNAPSHOT_TRACK="${TRAIN_EXECUTION_SCRIPT_TRACK:-${SCRIPT_DIR}/train_execution.sh}"
if [[ -f "${TRAIN_EXECUTION_SNAPSHOT_TRACK}" ]]; then
  cp "${TRAIN_EXECUTION_SNAPSHOT_TRACK}" "${RUN_META_DIR_TRACK}/train_execution_${RUN_STAMP_TRACK}.sh"
fi
{
  echo "timestamp=${RUN_STAMP_TRACK}"
  echo "checkpoint_root=${CHECKPOINT_ROOT_TRACK}"
  echo "model=${MODEL_NAME_TRACK}"
  echo "meta=${DATASET_META_NAME_TRACK}"
  echo "dataset_specs=${DATASET_SPECS_TRACK}"
  echo "dataset_root=${DATASET_NAME_TRACK}"
  echo "dataset_root_id_key=${DATASET_ROOT_ID_KEY_TRACK}"
  echo "input_mode=${INPUT_MODE_TRACK}"
  echo "num_processes=${NUM_PROCESSES_TRACK}"
  echo "mixed_precision=${MIXED_PRECISION_TRACK}"
  echo "checkpointing_steps=${CHECKPOINTING_STEPS_TRACK}"
  echo "learning_rate=${LEARNING_RATE_TRACK}"
  echo "adam_weight_decay=${ADAM_WEIGHT_DECAY_TRACK}"
  echo "adam_epsilon=${ADAM_EPSILON_TRACK}"
  echo "new_track_layers_lr=${NEW_TRACK_LAYERS_LR_TRACK}"
  echo "early_blocks_lr=${EARLY_BLOCKS_LR_TRACK}"
  echo "train_early_blocks=${TRAIN_EARLY_BLOCKS_TRACK}"
  echo "lr_warmup_steps=${LR_WARMUP_STEPS_TRACK}"
  echo "text_drop_ratio=${TEXT_DROP_RATIO_TRACK}"
  echo "resume_from_checkpoint=${RESUME_FROM_CHECKPOINT_TRACK}"
  echo "init_model_from_checkpoint=${INIT_MODEL_FROM_CHECKPOINT_TRACK}"
  echo "track_max_points=${TRACK_MAX_POINTS_TRACK}"
  echo "track_random_points_min=${TRACK_RANDOM_POINTS_MIN_TRACK}"
  echo "track_random_points_max=${TRACK_RANDOM_POINTS_MAX_TRACK}"
  echo "track_sort_selected_indices=${TRACK_SORT_SELECTED_INDICES_TRACK}"
  echo "track_point_id_mode=${TRACK_POINT_ID_MODE_TRACK}"
  echo "track_condition_drop_prob=${TRACK_CONDITION_DROP_PROB_TRACK}"
  echo "track_condition_mode=${TRACK_CONDITION_MODE_TRACK}"
  echo "apply_track_patch_embed_init=${APPLY_TRACK_PATCH_EMBED_INIT_TRACK}"
  echo "track_patch_init_mode=${TRACK_PATCH_INIT_MODE_TRACK}"
  echo "track_patch_init_gain=${TRACK_PATCH_INIT_GAIN_TRACK}"
  echo "track_latent_scale=${TRACK_LATENT_SCALE_TRACK}"
  echo "track_latent_first_frame_scale=${TRACK_LATENT_FIRST_FRAME_SCALE_TRACK:-<track_latent_scale>}"
  echo "track_latent_rest_frame_scale=${TRACK_LATENT_REST_FRAME_SCALE_TRACK:-<track_latent_scale>}"
  echo "add_track_init_noise=${ADD_TRACK_INIT_NOISE_TRACK}"
  echo "track_init_noise_scale=${TRACK_INIT_NOISE_SCALE_TRACK}"
  echo "track_head_hidden_dim=${TRACK_HEAD_HIDDEN_DIM_TRACK}"
  echo "first_frame_condition_drop_prob=${FIRST_FRAME_CONDITION_DROP_PROB_TRACK}"
  echo "validation_steps=${VALIDATION_STEPS_TRACK}"
  echo "validation_max_batches=${VALIDATION_MAX_BATCHES_TRACK}"
  echo "val_data_meta=${VAL_DATA_META_NAME_TRACK}"
  echo "val_data_dir=${VAL_DATASET_NAME_TRACK}"
  echo "new_params_only_steps=${NEW_PARAMS_ONLY_STEPS_TRACK}"
  echo "debug_weight_update=${DEBUG_WEIGHT_UPDATE_TRACK}"
  echo "debug_weight_update_topk=${DEBUG_WEIGHT_UPDATE_TOPK_TRACK}"
  echo "track_debug_vis_steps=${TRACK_DEBUG_VIS_STEPS_TRACK}"
  echo "track_debug_vis_dir=${TRACK_DEBUG_VIS_DIR_TRACK}"
  echo "track_debug_vis_sample_index=${TRACK_DEBUG_VIS_SAMPLE_INDEX_TRACK}"
  echo "track_debug_vis_max_frames=${TRACK_DEBUG_VIS_MAX_FRAMES_TRACK}"
  echo "track_debug_vis_max_points=${TRACK_DEBUG_VIS_MAX_POINTS_TRACK}"
  echo "track_debug_vis_fps=${TRACK_DEBUG_VIS_FPS_TRACK}"
  echo "num_train_epochs=${NUM_TRAIN_EPOCHS_TRACK}"
  echo "max_train_steps=${MAX_TRAIN_STEPS_TRACK}"
  echo "use_first_frame_condition=${USE_FIRST_FRAME_CONDITION_TRACK}"
  echo "precomputed_uncond_text_npz=${PRECOMPUTED_UNCOND_TEXT_NPZ_TRACK}"
} > "${RUN_META_DIR_TRACK}/run_args_${RUN_STAMP_TRACK}.txt"

"${PYTHON_BIN_TRACK}" -m accelerate.commands.launch \
  --num_processes="${NUM_PROCESSES_TRACK}" \
  --mixed_precision="${MIXED_PRECISION_TRACK}" \
  scripts/wan2.1_fun_track/train_track.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="${MODEL_NAME_TRACK}" \
  --train_data_dir="${DATASET_NAME_TRACK}" \
  --train_data_meta_track="${DATASET_META_NAME_TRACK}" \
  --input_mode_track="${INPUT_MODE_TRACK}" \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=${TRAIN_BATCH_SIZE_TRACK} \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs="${NUM_TRAIN_EPOCHS_TRACK}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS_TRACK}" \
  --validation_steps_track="${VALIDATION_STEPS_TRACK}" \
  --validation_max_batches_track="${VALIDATION_MAX_BATCHES_TRACK}" \
  --learning_rate="${LEARNING_RATE_TRACK}" \
  --new_track_layers_lr="${NEW_TRACK_LAYERS_LR_TRACK}" \
  --early_blocks_lr="${EARLY_BLOCKS_LR_TRACK}" \
  --train_early_blocks_track="${TRAIN_EARLY_BLOCKS_TRACK}" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps="${LR_WARMUP_STEPS_TRACK}" \
  --text_drop_ratio_track="${TEXT_DROP_RATIO_TRACK}" \
  --seed=42 \
  --report_to="${REPORT_TO_TRACK}" \
  --output_dir_track="${OUTPUT_DIR_TRACK}" \
  --mixed_precision="${MIXED_PRECISION_TRACK}" \
  --gradient_checkpointing="${GRADIENT_CHECKPOINTING_TRACK}" \
  --adam_weight_decay="${ADAM_WEIGHT_DECAY_TRACK}" \
  --adam_epsilon="${ADAM_EPSILON_TRACK}" \
  --max_grad_norm=1.0 \
  --train_mode="inpaint" \
  --trainable_modules_track "." \
  --new_params_only_steps_track="${NEW_PARAMS_ONLY_STEPS_TRACK}" \
  --track_debug_vis_steps="${TRACK_DEBUG_VIS_STEPS_TRACK}" \
  --track_debug_vis_dir="${TRACK_DEBUG_VIS_DIR_TRACK}" \
  --track_debug_vis_sample_index="${TRACK_DEBUG_VIS_SAMPLE_INDEX_TRACK}" \
  --track_debug_vis_max_frames="${TRACK_DEBUG_VIS_MAX_FRAMES_TRACK}" \
  --track_debug_vis_max_points="${TRACK_DEBUG_VIS_MAX_POINTS_TRACK}" \
  --track_debug_vis_fps="${TRACK_DEBUG_VIS_FPS_TRACK}" \
  --use_track_condition \
  --track_condition_mode="${TRACK_CONDITION_MODE_TRACK}" \
  --apply_track_patch_embed_init_track="${APPLY_TRACK_PATCH_EMBED_INIT_TRACK}" \
  --track_patch_init_mode="${TRACK_PATCH_INIT_MODE_TRACK}" \
  --track_patch_init_gain="${TRACK_PATCH_INIT_GAIN_TRACK}" \
  --track_latent_scale="${TRACK_LATENT_SCALE_TRACK}" \
  --add_track_init_noise="${ADD_TRACK_INIT_NOISE_TRACK}" \
  --track_init_noise_scale="${TRACK_INIT_NOISE_SCALE_TRACK}" \
  --track_max_points="${TRACK_MAX_POINTS_TRACK}" \
  --track_random_points_min="${TRACK_RANDOM_POINTS_MIN_TRACK}" \
  --track_random_points_max="${TRACK_RANDOM_POINTS_MAX_TRACK}" \
  --track_sort_selected_indices="${TRACK_SORT_SELECTED_INDICES_TRACK}" \
  --track_point_id_mode="${TRACK_POINT_ID_MODE_TRACK}" \
  --track_condition_drop_prob="${TRACK_CONDITION_DROP_PROB_TRACK}" \
  --first_frame_condition_drop_prob_track="${FIRST_FRAME_CONDITION_DROP_PROB_TRACK}" \
  --track_normalize \
  "${EXTRA_ARGS_TRACK[@]}"
