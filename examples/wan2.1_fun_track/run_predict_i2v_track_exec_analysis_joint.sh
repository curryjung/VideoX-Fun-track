#!/usr/bin/env bash
set -euo pipefail

# Metadata-based analysis loop (joint text-motion guidance)
VAL_METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/val_metadata_track.json"
TRAIN_DATA_DIR="/data/shared-vilab/datasets/OpenVid-1M"
# SAMPLE_INDEX_LIST=(1 0 2 3 4 5 6 7 8 9)
SAMPLE_INDEX_LIST=(9 8 7 6)
SAMPLE_INDEX_LIST=(10 11 12 13 14 15 16 17 18 19)
SAMPLE_INDEX_LIST=(20 21 22 23 24 25 26 27 28 29)
SAMPLE_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
# SAMPLE_INDEX_LIST=(26 27 5 3 2 1 16 20)
TRACK_LATENT_SCALE_LIST=(1.0)

# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_normalize_fixed_Openvid-0p6M_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-1200"
# SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_normalize_fixed_Openvid-0p6M_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-1200/analysis_from_val_metadata_joint"
TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track-alpha_0p5_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-4000"
SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/wan_track_track-patch-embed-init-track-alpha_0p5_fineMotion12800_dropout_first-frame_0p1_text_0p1_track_0p1/checkpoint-4000/analysis_from_val_metadata_joint"

GUIDANCE_MODE="${GUIDANCE_MODE:-joint_tm}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.0}"
# Keep text/first-frame on SAMPLE_INDEX, but shift track row by this offset for mismatch tests.
TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET:-0}"

ASSET_BASE_DIR="${ASSET_BASE_DIR:-}"
if [[ -n "${ASSET_BASE_DIR}" ]]; then
    if [[ ! -f "${ASSET_BASE_DIR}/first_frame.png" ]]; then
        echo "[error] missing first frame: ${ASSET_BASE_DIR}/first_frame.png"
        exit 1
    fi
    if [[ ! -f "${ASSET_BASE_DIR}/transformed_tracks_grid50_survived.npz" ]]; then
        echo "[error] missing track npz: ${ASSET_BASE_DIR}/transformed_tracks_grid50_survived.npz"
        exit 1
    fi

    PROMPT="${PROMPT:-}"
    if [[ -z "${PROMPT}" && -f "${ASSET_BASE_DIR}/image_caption.txt" ]]; then
        PROMPT="$(awk 'f{print} /^$/{f=1}' "${ASSET_BASE_DIR}/image_caption.txt" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
    fi
    if [[ -z "${PROMPT}" ]]; then
        echo "[error] prompt is empty. Set PROMPT or add image_caption.txt with caption body."
        exit 1
    fi

    TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE:-1.0}"
    ASSET_NAME="${ASSET_NAME:-$(basename "${ASSET_BASE_DIR}")}"
    SAVE_DIR="${SAVE_BASE_DIR}/${ASSET_NAME}/scale_${TRACK_LATENT_SCALE}"
    mkdir -p "${SAVE_DIR}"
    DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
    TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"
    OUTPUT_NAME_SUFFIX="jointtm_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_p2000_s${TRACK_LATENT_SCALE}_toff${TRACK_CONDITION_INDEX_OFFSET}"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
    SAVE_DIR="${SAVE_DIR}" \
    PROMPT="${PROMPT}" \
    VALIDATION_IMAGE_START="${ASSET_BASE_DIR}/first_frame.png" \
    TRACK_FILE_PATH="${ASSET_BASE_DIR}/transformed_tracks_grid50_survived.npz" \
    TRACK_NORMALIZE="${TRACK_NORMALIZE:-false}" \
    COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
    OVERLAY_PAD_VALUE=0 \
    OVERLAY_LINEWIDTH=1 \
    OVERLAY_TRACE_FRAMES=8 \
    TRACK_MAX_POINTS=2000 \
    TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE}" \
    TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET}" \
    DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
    TRACK_ANALYSIS="${TRACK_ANALYSIS}" \
    GUIDANCE_MODE="${GUIDANCE_MODE}" \
    TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
    MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
    OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX}" \
    bash examples/wan2.1_fun_track/run_predict_i2v_track_joint.sh
    exit 0
fi

for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
    for TRACK_LATENT_SCALE in "${TRACK_LATENT_SCALE_LIST[@]}"; do
        SAVE_DIR="${SAVE_BASE_DIR}/sample_idx_${SAMPLE_INDEX}/scale_${TRACK_LATENT_SCALE}"
        mkdir -p "${SAVE_DIR}"
        DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
        TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"
        OUTPUT_NAME_SUFFIX="jointtm_wt${TEXT_GUIDANCE_WEIGHT}_wm${MOTION_GUIDANCE_WEIGHT}_p2000_s${TRACK_LATENT_SCALE}_toff${TRACK_CONDITION_INDEX_OFFSET}"
        CUDA_VISIBLE_DEVICES=0 \
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
        TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET}" \
        DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
        TRACK_ANALYSIS="${TRACK_ANALYSIS}" \
        GUIDANCE_MODE="${GUIDANCE_MODE}" \
        TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
        MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
        bash examples/wan2.1_fun_track/run_predict_i2v_track_joint.sh
    done
done
