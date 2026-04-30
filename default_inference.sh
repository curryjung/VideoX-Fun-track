WAN_PATCH_GROUP_ANALYSIS="${WAN_PATCH_GROUP_ANALYSIS:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" \
torchrun --nproc-per-node=1 examples/wan2.1_fun/predict_i2v.py