TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE:-4.0}"
TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track-alpha_0p5_fork_deft-sunset-41-4000ckpt_dropout_0p1s_OpenVid-1M_550000/checkpoint-5400"
TRANSFORMER_CHECKPOINT_PATH="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints/wan_track_track-patch-embed-init-track_bs16_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_256_dropout_first-frame_0p1_text_0p1_track_0p1"

# SAMPLE_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/asset/cute_shiba"
SAMPLE_BASE_DIR_BASE="/data/project-vilab/jaeseok/VideoX-Fun/asset/track_samples"
SAVE_BASE_DIR_BASE="/data/project-vilab/jaeseok/VideoX-Fun/samples/wan-videos-fun-i2v-track"
CHECKPOINT_BASE_DIR="/data/project-vilab/jaeseok/VideoX-Fun/checkpoints"


SAMPLE_INDEX_LIST=(0 1 2 3 4 5)
# SAMPLE_INDEX_LIST=(0 1)
SAMPLE_NAME_LIST=(cute_shiba2 cute_shiba3 cute_shiba4 cute_shiba5 cute_shiba6 cute_shiba7 cute_shiba8 cute_shiba9)
SAMPLE_NAME_LIST=(cute_shiba10 cute_shiba3 cute_shiba4 cute_shiba5 cute_shiba6 cute_shiba7 cute_shiba8 cute_shiba9)
SAMPLE_NAME_LIST=(cute_shiba11 cute_shiba12)
SAMPLE_NAME_LIST=(dragon_riding1 dragon_riding2 dragon_riding3 dragon_riding4 dragon_riding5)
# SAMPLE_NAME_LIST=(cute_shiba12 cute_shiba11)
# SAMPLE_NAME_LIST=(a_medieval_woman_painting1 a_medieval_woman_painting2 a_medieval_woman_painting3 a_medieval_woman_painting4 a_medieval_woman_painting5 a_medieval_woman_painting6)
ckpt_list=(1000)
SEED_LIST=(42 41 40)


EXP_NAME="wan_track_patch-copy_first_gain-1.0_scale-4.0_track_local-point-id_bs64_train_78k_h_dim_64_all-dropouts_0p1"
# EXP_NAME="wan_track_track-patch-embed-init-track_local-point-id-mode_bs64_alpha_1p0_init_noise_0p01_bin8_train_78k_h_dim_64_dropout_first-frame_0p1_text_0p1_track_0p1"

PROMPT_list=("A close-up of a Shiba Inu dog sitting on a beige couch. The dog is facing the camera and is looking directly at the camera with a curious expression. Its fur is light brown and fluffy, and its eyes are dark and alert." \
"A close-up of a Shiba Inu dog sitting on a beige couch. The dog is facing the camera and is looking directly at the camera with a curious expression. Its fur is light brown and fluffy, and its eyes are dark and alert." \
"A close-up of a Shiba Inu dog sitting on a beige couch. The dog is facing the camera and is looking directly at the camera with a curious expression. Its fur is light brown and fluffy, and its eyes are dark and alert." \
"A close-up of a Shiba Inu dog sitting on a beige couch. The dog is facing the camera and is looking directly at the camera with a curious expression. Its fur is light brown and fluffy, and its eyes are dark and alert.")

PROMPT_list=("A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog" \
"A Shiba Inu dog"
)

# PROMPT_list=("A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# "A noble young woman in Renaissance royal clothing" \
# )

# PROMPT_list=("A locked-off medium-close shot of a Shiba Inu sitting indoors on a soft couch, subtly moving its head while the camera stays completely still. " \
# "A locked-off medium-close shot of a Shiba Inu sitting indoors on a soft couch, subtly moving its head while the camera stays completely still. " \
# "A locked-off medium-close shot of a Shiba Inu sitting indoors on a soft couch, subtly moving its head while the camera stays completely still. " \
# "A locked-off medium-close shot of a Shiba Inu sitting indoors on a soft couch, subtly moving its head while the camera stays completely still. ")

GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
# Wan-Move style unified CFG: [text+motion] vs [null_text+null_motion].
# GUIDANCE_MODE="${GUIDANCE_MODE:-unified}"
GUIDANCE_MODE="${GUIDANCE_MODE:-motion_only}"
TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT:-3.0}"
MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT:-3.0}"
TRACK_NORMALIZE="${TRACK_NORMALIZE:-true}"
TRACK_NORMALIZE_HEIGHT="${TRACK_NORMALIZE_HEIGHT:-480}"
TRACK_NORMALIZE_WIDTH="${TRACK_NORMALIZE_WIDTH:-832}"
TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM:-64}"
TRACK_MAX_POINTS="${TRACK_MAX_POINTS:-2000}"
TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE:-random}"
TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES:-false}"
TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED:-42}"
TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE:-local}"
TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET:-0}"

# Track normalization visualization options.
VISUALIZE_TRACK_NORMALIZATION="${VISUALIZE_TRACK_NORMALIZATION:-true}"
VIS_TRACK_MODES="${VIS_TRACK_MODES:-auto normalized pixel}"
VIS_TRACK_SAVE_MP4="${VIS_TRACK_SAVE_MP4:-false}"
VIS_TRACK_MAX_FRAMES="${VIS_TRACK_MAX_FRAMES:-32}"
VIS_TRACK_FPS="${VIS_TRACK_FPS:-8}"
VIS_TRACK_HEAT_PERCENTILE="${VIS_TRACK_HEAT_PERCENTILE:-99.0}"
VIS_TRACK_HEAT_GAMMA="${VIS_TRACK_HEAT_GAMMA:-0.55}"

DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-}" \


for ckpt in "${ckpt_list[@]}"; do
    TRANSFORMER_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}/checkpoint-${ckpt}"
    SAVE_BASE_DIR="${SAVE_BASE_DIR_BASE}/${EXP_NAME}/checkpoint-${ckpt}"
    for SAMPLE_INDEX in "${SAMPLE_INDEX_LIST[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
            SAMPLE_NAME="${SAMPLE_NAME_LIST[${SAMPLE_INDEX}]}"
            SAVE_DIR="${SAVE_BASE_DIR}/${SAMPLE_NAME}/${GUIDANCE_MODE}_guidance_${GUIDANCE_SCALE}_text_${TEXT_GUIDANCE_WEIGHT}_motion_${MOTION_GUIDANCE_WEIGHT}_local"
            SAMPLE_BASE_DIR="${SAMPLE_BASE_DIR_BASE}/${SAMPLE_NAME}"
            mkdir -p "${SAVE_DIR}"
            DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION:-true}"
            TRACK_ANALYSIS="${TRACK_ANALYSIS:-true}"
            TRACK_POINTS_SUFFIX="pall"

            CUDA_VISIBLE_DEVICES=0 \
            TRANSFORMER_CHECKPOINT_PATH="${TRANSFORMER_CHECKPOINT_PATH}" \
            SAVE_DIR="${SAVE_DIR}" \
            PROMPT="${PROMPT_list[${SAMPLE_INDEX}]}" \
            VALIDATION_IMAGE_START="${SAMPLE_BASE_DIR}/first_frame.png" \
            TRACK_FILE_PATH="${SAMPLE_BASE_DIR}/transformed_tracks_grid50_survived.npz" \
            COTRACKER_ROOT="/data/project-vilab/jaeseok/co-tracker" \
            OVERLAY_PAD_VALUE=0 \
            OVERLAY_LINEWIDTH=1 \
            OVERLAY_TRACE_FRAMES=8 \
            TRACK_MAX_POINTS="${TRACK_MAX_POINTS}" \
            TRACK_NORMALIZE="${TRACK_NORMALIZE}" \
            TRACK_HEAD_HIDDEN_DIM="${TRACK_HEAD_HIDDEN_DIM}" \
            TRACK_LATENT_SCALE="${TRACK_LATENT_SCALE}" \
            TRACK_CONDITION_INDEX_OFFSET="${TRACK_CONDITION_INDEX_OFFSET}" \
            TRACK_POINT_SAMPLE_MODE="${TRACK_POINT_SAMPLE_MODE}" \
            TRACK_SORT_SELECTED_INDICES="${TRACK_SORT_SELECTED_INDICES}" \
            TRACK_POINT_ID_MODE="${TRACK_POINT_ID_MODE}" \
            TRACK_POINT_SAMPLE_SEED="${TRACK_POINT_SAMPLE_SEED}" \
            SEED="${SEED}" \
            ZERO_CLIP_CONTEXT="${ZERO_CLIP_CONTEXT}" \
            DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
            TRACK_ANALYSIS="${TRACK_ANALYSIS}" \
            GUIDANCE_MODE="${GUIDANCE_MODE}" \
            GUIDANCE_SCALE="${GUIDANCE_SCALE}" \
            TEXT_GUIDANCE_WEIGHT="${TEXT_GUIDANCE_WEIGHT}" \
            MOTION_GUIDANCE_WEIGHT="${MOTION_GUIDANCE_WEIGHT}" \
            DEBUG_TRACK_CONDITION="${DEBUG_TRACK_CONDITION}" \
            bash examples/wan2.1_fun_track/run_predict_i2v_track_joint.sh
        done
    done
done

# bash examples/wan2.1_fun_track/run_predict_i2v_track_joint.sh
# bash examples/wan2.1_fun_track/run_predict_i2v_track_text_only.sh
