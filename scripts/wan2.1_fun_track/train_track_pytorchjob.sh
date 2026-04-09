#!/usr/bin/env bash
# PyTorchJob-friendly launcher for Wan2.1 track training.
# - Keeps training logic in train_track.py unchanged.
# - Accepts hyperparameters via env vars (no image rebuild needed).
# - Supports Kubeflow PET_* rendezvous envs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHON_BIN_TRACK="${PYTHON_BIN_TRACK:-python}"
export TORCHRUN_BIN_TRACK="${TORCHRUN_BIN_TRACK:-torchrun}"

# Dist config: prefer PET_* from PyTorchJob, fallback to local envs.
export NUM_PROCESSES_TRACK="${NUM_PROCESSES_TRACK:-${PET_NPROC_PER_NODE:-1}}"
export NUM_MACHINES_TRACK="${NUM_MACHINES_TRACK:-${PET_NNODES:-1}}"
export RDZV_ID_TRACK="${RDZV_ID_TRACK:-${PET_RDZV_ID:-wan2-track}}"
export RDZV_BACKEND_TRACK="${RDZV_BACKEND_TRACK:-${PET_RDZV_BACKEND:-c10d}}"
export RDZV_ENDPOINT_TRACK="${RDZV_ENDPOINT_TRACK:-${PET_RDZV_ENDPOINT:-127.0.0.1:29500}}"

export MIXED_PRECISION_TRACK="${MIXED_PRECISION_TRACK:-bf16}"
export GRADIENT_CHECKPOINTING_TRACK="${GRADIENT_CHECKPOINTING_TRACK:-false}"

if ! "${PYTHON_BIN_TRACK}" -c "import videox_fun" 2>/dev/null; then
  echo "[info] videox_fun not found. Installing editable package from PVC path..."
  "${PYTHON_BIN_TRACK}" -m pip install -e /data/project-vilab/jaeseok/VideoX-Fun
fi

if ! "${PYTHON_BIN_TRACK}" -c "import videox_fun" 2>/dev/null; then
  echo "[error] Cannot import videox_fun after editable install."
  echo "  attempted: ${PYTHON_BIN_TRACK} -m pip install -e /data/project-vilab/jaeseok/VideoX-Fun"
  exit 1
fi

export MODEL_NAME_TRACK="${MODEL_NAME_TRACK:-models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP}"
export DATASET_NAME_TRACK="${DATASET_NAME_TRACK:-datasets/internal_datasets/}"
export DATASET_META_NAME_TRACK="${DATASET_META_NAME_TRACK:-datasets/internal_datasets/metadata_track.json}"
export INPUT_MODE_TRACK="${INPUT_MODE_TRACK:-latent}"
export DATASET_ROOT_MAP_JSON_TRACK="${DATASET_ROOT_MAP_JSON_TRACK:-}"
export DATASET_ROOT_ID_KEY_TRACK="${DATASET_ROOT_ID_KEY_TRACK:-root_id}"
export TRACK_MAX_POINTS_TRACK="${TRACK_MAX_POINTS_TRACK:--1}"
export TRACK_RANDOM_POINTS_MIN_TRACK="${TRACK_RANDOM_POINTS_MIN_TRACK:-1000}"
export TRACK_RANDOM_POINTS_MAX_TRACK="${TRACK_RANDOM_POINTS_MAX_TRACK:-2500}"
export TRACK_CONDITION_DROP_PROB_TRACK="${TRACK_CONDITION_DROP_PROB_TRACK:-0.0}"
export VAL_DATA_META_NAME_TRACK="${VAL_DATA_META_NAME_TRACK:-}"
export VAL_DATASET_NAME_TRACK="${VAL_DATASET_NAME_TRACK:-}"
export VALIDATION_STEPS_TRACK="${VALIDATION_STEPS_TRACK:-0}"
export VALIDATION_MAX_BATCHES_TRACK="${VALIDATION_MAX_BATCHES_TRACK:-8}"
export NEW_PARAMS_ONLY_STEPS_TRACK="${NEW_PARAMS_ONLY_STEPS_TRACK:-0}"
export DEBUG_WEIGHT_UPDATE_TRACK="${DEBUG_WEIGHT_UPDATE_TRACK:-false}"
export DEBUG_WEIGHT_UPDATE_TOPK_TRACK="${DEBUG_WEIGHT_UPDATE_TOPK_TRACK:-30}"
export CHECKPOINTING_STEPS_TRACK="${CHECKPOINTING_STEPS_TRACK:-500}"
export LEARNING_RATE_TRACK="${LEARNING_RATE_TRACK:-1e-5}"
export LR_WARMUP_STEPS_TRACK="${LR_WARMUP_STEPS_TRACK:-800}"
export USE_FIRST_FRAME_CONDITION_TRACK="${USE_FIRST_FRAME_CONDITION_TRACK:-true}"

export REPORT_TO_TRACK="${REPORT_TO_TRACK:-wandb}"
export WANDB_RUN_NAME_TRACK="${WANDB_RUN_NAME_TRACK:-}"

export OUTPUT_DIR_TRACK="${OUTPUT_DIR_TRACK:-output_dir_wan2.1_fun_track}"
export CHECKPOINT_DIR_TRACK="${CHECKPOINT_DIR_TRACK:-}"
export RESUME_FROM_CHECKPOINT_TRACK="${RESUME_FROM_CHECKPOINT_TRACK:-}"
export INIT_MODEL_FROM_CHECKPOINT_TRACK="${INIT_MODEL_FROM_CHECKPOINT_TRACK:-}"

if [[ ! -f "${DATASET_META_NAME_TRACK}" ]]; then
  echo "[error] Metadata file not found: ${DATASET_META_NAME_TRACK}"
  echo "[hint] Build it first with:"
  echo "  python scripts/wan2.1_fun_track/build_metadata_track.py --help"
  exit 1
fi

EXTRA_ARGS_TRACK=()
if [[ -n "${DATASET_ROOT_MAP_JSON_TRACK}" ]]; then
  EXTRA_ARGS_TRACK+=("--train_data_root_map_json_track=${DATASET_ROOT_MAP_JSON_TRACK}")
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

CHECKPOINT_ROOT_TRACK="${CHECKPOINT_DIR_TRACK:-${OUTPUT_DIR_TRACK}}"
RUN_META_DIR_TRACK="${CHECKPOINT_ROOT_TRACK}/run_meta"
mkdir -p "${RUN_META_DIR_TRACK}"
RUN_STAMP_TRACK="$(date +%Y%m%d_%H%M%S)"
cp "${BASH_SOURCE[0]}" "${RUN_META_DIR_TRACK}/train_track_pytorchjob_${RUN_STAMP_TRACK}.sh"
if [[ -f "${SCRIPT_DIR}/train_execution_pytorchjob.sh" ]]; then
  cp "${SCRIPT_DIR}/train_execution_pytorchjob.sh" "${RUN_META_DIR_TRACK}/train_execution_pytorchjob_${RUN_STAMP_TRACK}.sh"
fi
{
  echo "timestamp=${RUN_STAMP_TRACK}"
  echo "checkpoint_root=${CHECKPOINT_ROOT_TRACK}"
  echo "model=${MODEL_NAME_TRACK}"
  echo "meta=${DATASET_META_NAME_TRACK}"
  echo "input_mode=${INPUT_MODE_TRACK}"
  echo "num_processes_per_node=${NUM_PROCESSES_TRACK}"
  echo "num_machines=${NUM_MACHINES_TRACK}"
  echo "rdzv_id=${RDZV_ID_TRACK}"
  echo "rdzv_backend=${RDZV_BACKEND_TRACK}"
  echo "rdzv_endpoint=${RDZV_ENDPOINT_TRACK}"
  echo "mixed_precision=${MIXED_PRECISION_TRACK}"
  echo "checkpointing_steps=${CHECKPOINTING_STEPS_TRACK}"
  echo "learning_rate=${LEARNING_RATE_TRACK}"
  echo "lr_warmup_steps=${LR_WARMUP_STEPS_TRACK}"
  echo "init_model_from_checkpoint=${INIT_MODEL_FROM_CHECKPOINT_TRACK}"
  echo "track_max_points=${TRACK_MAX_POINTS_TRACK}"
  echo "track_random_points_min=${TRACK_RANDOM_POINTS_MIN_TRACK}"
  echo "track_random_points_max=${TRACK_RANDOM_POINTS_MAX_TRACK}"
  echo "track_condition_drop_prob=${TRACK_CONDITION_DROP_PROB_TRACK}"
  echo "validation_steps=${VALIDATION_STEPS_TRACK}"
  echo "validation_max_batches=${VALIDATION_MAX_BATCHES_TRACK}"
  echo "val_data_meta=${VAL_DATA_META_NAME_TRACK}"
  echo "val_data_dir=${VAL_DATASET_NAME_TRACK}"
  echo "new_params_only_steps=${NEW_PARAMS_ONLY_STEPS_TRACK}"
  echo "debug_weight_update=${DEBUG_WEIGHT_UPDATE_TRACK}"
  echo "debug_weight_update_topk=${DEBUG_WEIGHT_UPDATE_TOPK_TRACK}"
  echo "use_first_frame_condition=${USE_FIRST_FRAME_CONDITION_TRACK}"
} > "${RUN_META_DIR_TRACK}/run_args_${RUN_STAMP_TRACK}.txt"

{
  echo "timestamp=${RUN_STAMP_TRACK}"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python_bin=${PYTHON_BIN_TRACK}"
  echo "torchrun_bin=${TORCHRUN_BIN_TRACK}"
  echo "pet_nnodes=${PET_NNODES:-}"
  echo "pet_nproc_per_node=${PET_NPROC_PER_NODE:-}"
  echo "pet_rdzv_id=${PET_RDZV_ID:-}"
  echo "pet_rdzv_backend=${PET_RDZV_BACKEND:-}"
  echo "pet_rdzv_endpoint=${PET_RDZV_ENDPOINT:-}"
  echo "nccl_debug=${NCCL_DEBUG:-}"
  echo "torch_distributed_debug=${TORCH_DISTRIBUTED_DEBUG:-}"
  echo "torch_cpp_log_level=${TORCH_CPP_LOG_LEVEL:-}"
  echo "gloo_socket_family=${GLOO_SOCKET_FAMILY:-}"
  echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "master_addr=${MASTER_ADDR:-}"
  echo "master_port=${MASTER_PORT:-}"
} > "${RUN_META_DIR_TRACK}/run_env_${RUN_STAMP_TRACK}.txt"

TORCHRUN_ARGS=(
  "--nnodes=${NUM_MACHINES_TRACK}"
  "--nproc_per_node=${NUM_PROCESSES_TRACK}"
)

if [[ "${NUM_MACHINES_TRACK}" -gt 1 ]]; then
  TORCHRUN_ARGS+=(
    "--rdzv_id=${RDZV_ID_TRACK}"
    "--rdzv_backend=${RDZV_BACKEND_TRACK}"
    "--rdzv_endpoint=${RDZV_ENDPOINT_TRACK}"
  )
fi

"${TORCHRUN_BIN_TRACK}" \
  "${TORCHRUN_ARGS[@]}" \
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
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps="${CHECKPOINTING_STEPS_TRACK}" \
  --validation_steps_track="${VALIDATION_STEPS_TRACK}" \
  --validation_max_batches_track="${VALIDATION_MAX_BATCHES_TRACK}" \
  --learning_rate="${LEARNING_RATE_TRACK}" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps="${LR_WARMUP_STEPS_TRACK}" \
  --seed=42 \
  --report_to="${REPORT_TO_TRACK}" \
  --output_dir_track="${OUTPUT_DIR_TRACK}" \
  --mixed_precision="${MIXED_PRECISION_TRACK}" \
  --gradient_checkpointing="${GRADIENT_CHECKPOINTING_TRACK}" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=1.0 \
  --train_mode="inpaint" \
  --trainable_modules_track "." \
  --new_params_only_steps_track="${NEW_PARAMS_ONLY_STEPS_TRACK}" \
  --use_track_condition \
  --track_max_points="${TRACK_MAX_POINTS_TRACK}" \
  --track_random_points_min="${TRACK_RANDOM_POINTS_MIN_TRACK}" \
  --track_random_points_max="${TRACK_RANDOM_POINTS_MAX_TRACK}" \
  --track_condition_drop_prob="${TRACK_CONDITION_DROP_PROB_TRACK}" \
  --track_normalize \
  "${EXTRA_ARGS_TRACK[@]}"
