#!/usr/bin/env bash
set -euo pipefail

cd /data/project-vilab/jaeseok/VideoX-Fun

# Fixed sample setup
SAMPLE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/asset/--oP9WWSIas_0_604to1168/processed_832x480_fps16"
CHECKPOINT_ROOT="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_normalize_first_frame_fix"
SAVE_ROOT="samples/wan-videos-fun-i2v-track/ckpt_compare"

# Pick checkpoints to compare
CHECKPOINT_STEPS=(200 400 600 800)

for step in "${CHECKPOINT_STEPS[@]}"; do
  CKPT="${CHECKPOINT_ROOT}/checkpoint-${step}"

  for mode in on off fake; do
    EXTRA_ENV=()
    SUFFIX="ckpt${step}_${mode}"

    if [[ "${mode}" == "off" ]]; then
      EXTRA_ENV+=(FORCE_TRACK_CONDITION_NONE=true)
    elif [[ "${mode}" == "fake" ]]; then
      EXTRA_ENV+=(RANDOM_FAKE_TRACK=true)
    fi

    echo "=== checkpoint ${step} / mode ${mode} ==="

    env \
      CUDA_VISIBLE_DEVICES=1 \
      TRANSFORMER_CHECKPOINT_PATH="${CKPT}" \
      VALIDATION_IMAGE_START="${SAMPLE_BASE_DIR}/first_frame.png" \
      TRACK_FILE_PATH="${SAMPLE_BASE_DIR}/transformed_tracks_grid50_survived.npz" \
      TEXT_FEATURE_PATH="${SAMPLE_BASE_DIR}/text_feature_wan_t5.npz" \
      CLIP_FEATURE_PATH="${SAMPLE_BASE_DIR}/first_frame_clip_feature.npz" \
      COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
      OVERLAY_PAD_VALUE=0 \
      OVERLAY_LINEWIDTH=1 \
      OVERLAY_TRACE_FRAMES=8 \
      TRACK_NORMALIZE=true \
      DEBUG_TRACK_CONDITION=true \
      OUTPUT_NAME_SUFFIX="${SUFFIX}" \
      SAVE_DIR="${SAVE_ROOT}/checkpoint-${step}" \
      "${EXTRA_ENV[@]}" \
      bash examples/wan2.1_fun_track/run_predict_i2v_track.sh \
      2>&1 | tee "${SAVE_ROOT}/checkpoint-${step}/${SUFFIX}.log"
  done
done