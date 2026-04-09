model_name="Wan2.1-Fun-V1.1-1.3B-InP"

video_x_fun_save_dir="/data/shared-vilab/pretrained_models/Wan2.1-Fun-1.3B-InP"
video_x_fun_save_dir="/data/shared-vilab/pretrained_models/$model_name"

huggingface-cli download alibaba-pai/$model_name --local-dir $video_x_fun_save_dir
mkdir -p models/Diffusion_Transformer
cd models/Diffusion_Transformer 
ln -s /data/shared-vilab/pretrained_models/$model_name .