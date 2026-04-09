# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/output_dir_wan2.1_fun_track/checkpoint-1750" \
# METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track.json" \
# TRAIN_DATA_DIR="/data/shared-vilab/datasets/OpenVid-1M" \
# SAMPLE_INDEX=0 \
# USE_PROMPT_FROM_METADATA=true \
# bash examples/wan2.1_fun_track/run_predict_i2v_track.sh


# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/output_dir_wan2.1_fun_track/checkpoint-1750" \
# METADATA_PATH="/data/project-vilab/jaeseok/VideoX-Fun/datasets/internal_datasets/metadata_track.json" \
# TRAIN_DATA_DIR="/data/shared-vilab/datasets/OpenVid-1M/out_preprocess_openvid_cotracker_preshard_20260402_134900" \
# SAMPLE_INDEX=0 \
# USE_PROMPT_FROM_METADATA=true \
# bash examples/wan2.1_fun_track/run_predict_i2v_track.sh


# SAMPLE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/asset/cute_shiba"
# SAVE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/many_changes_only_new_atfirst_1000_steps_2e-5_ckpt2000/cute_shiba"
# PROMPT="The image is a close-up of a Shiba Inu dog sitting on a beige couch. The dog is facing the camera and is looking directly at the camera with a curious expression. Its fur is light brown and fluffy, and its eyes are dark and alert. The background is blurred, but it appears to be a living room with a plant and a picture frame on the wall."
# DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
# CUDA_VISIBLE_DEVICES=1 \
# TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_NEW_PARAMS_ONLY_STEPS_1000/checkpoint-2000" \
# SAVE_DIR="${SAVE_DIR}" \
# PROMPT="${PROMPT}" \
# VALIDATION_IMAGE_START="${SAMPLE_BASE_DIR}/first_frame.png" \
# TRACK_FILE_PATH="${SAMPLE_BASE_DIR}/transformed_tracks_grid50_survived.npz" \
# COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
# OVERLAY_PAD_VALUE=0 \
# OVERLAY_LINEWIDTH=1 \
# OVERLAY_TRACE_FRAMES=8 \
# OVERLAY_MAX_POINTS=-1 \
# TRACK_MAX_POINTS=300 \
# DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
# bash examples/wan2.1_fun_track/run_predict_i2v_track.sh


# In training dataset
SAMPLE_NAME_LIST=("--OoKiF0bUE_19_950to1091" "--VX0u52J5Y_37_0to118" "----oP9WWSIas_0_604to1168")
SAVE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track/many_changes_only_new_atfirst_1000_steps_2e-5_ckpt2200"

for SAMPLE_NAME in "${SAMPLE_NAME_LIST[@]}"; do
    SAMPLE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/asset/${SAMPLE_NAME}/processed_832x480_fps16"
    SAVE_DIR="${SAVE_BASE_DIR}/${SAMPLE_NAME}"
    mkdir -p "${SAVE_DIR}"
    DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
    CUDA_VISIBLE_DEVICES=5 \
    TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_NEW_PARAMS_ONLY_STEPS_1000/checkpoint-2200" \
    VALIDATION_IMAGE_START="${SAMPLE_BASE_DIR}/first_frame.png" \
    TRACK_FILE_PATH="${SAMPLE_BASE_DIR}/transformed_tracks_grid50_survived.npz" \
    TEXT_FEATURE_PATH="${SAMPLE_BASE_DIR}/text_feature_wan_t5.npz" \
    CLIP_FEATURE_PATH="${SAMPLE_BASE_DIR}/first_frame_clip_feature.npz" \
    COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
    FORCE_TRACK_CONDITION_NONE=true \
    SAVE_DIR="${SAVE_DIR}" \
    OVERLAY_PAD_VALUE=0 \
    OVERLAY_LINEWIDTH=1 \
    OVERLAY_TRACE_FRAMES=8 \
    OVERLAY_MAX_POINTS=-1 \
    TRACK_MAX_POINTS=2000 \
    DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
    bash examples/wan2.1_fun_track/run_predict_i2v_track.sh
done