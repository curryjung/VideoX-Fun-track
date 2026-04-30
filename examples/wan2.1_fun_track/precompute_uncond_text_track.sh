python scripts/wan2.1_fun_track/precompute_uncond_text_track.py \
  --config_path=config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path=models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP \
  --output_npz=/data/project-vilab/jaeseok/VideoX-Fun/asset/t5_uncond_empty_prompt.npz \
  --tokenizer_max_length=512 \
  --prompt=""