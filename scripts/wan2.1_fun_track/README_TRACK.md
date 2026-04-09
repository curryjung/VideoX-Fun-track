# Wan2.1 I2V Track Training Scaffold

This directory contains phase-1 training scaffold code for Wan2.1 i2v + track
conditioning.

## Defaults for pretrained i2v initialization

- `--config_path config/wan2.1/wan_civitai.yaml`
- `--pretrained_model_name_or_path models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP`

The scaffold initializes:
- tokenizer/text encoder
- VAE
- `WanTransformer3DModelTrack` (inherits Wan i2v transformer)

## Metadata contract

Use JSON/CSV metadata with keys:
- `file_path`
- `text`
- `type` (`video` or `image`)
- `track_file_path` (new)

The `track_file_path` should point to an `.npz` file containing:
- `tracks_compressed` (preferred) or `tracks`
- `visibility_compressed` (preferred) or `visibility`

## Dry-run command (1 batch / 1 step)

```bash
accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun_track/train_track.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP" \
  --train_data_dir "/path/to/data_root" \
  --train_data_meta_track "/path/to/metadata_track.json" \
  --train_mode "inpaint" \
  --train_batch_size 1 \
  --num_train_epochs 1 \
  --max_train_steps 1 \
  --use_track_condition \
  --dry_run_track
```

## TODO (modeling phase)

`WanTransformer3DModelTrack.forward(..., track_condition=...)` currently accepts
the condition payload and ignores it. Integrating track features into attention
blocks is intentionally left for phase-2 modeling work.

## Build metadata_track.json from preprocess outputs

```bash
python scripts/wan2.1_fun_track/build_metadata_track.py \
  --preprocess_root "/data/shared-vilab/datasets/OpenVid-1M/out_preprocess_openvid_cotracker_preshard_20260402_13490" \
  --output_meta "datasets/internal_datasets/metadata_track.json" \
  --sample_media latent
```

Multiple preprocess roots (from several servers):

```bash
python scripts/wan2.1_fun_track/build_metadata_track.py \
  --preprocess_root "/path/serverA_out" \
  --preprocess_root "/path/serverB_out" \
  --preprocess_root "/path/serverC_out" \
  --output_meta "datasets/internal_datasets/metadata_track.json" \
  --update_existing
```

Fast mode notes:
- default `--discovery_mode fixed` (recommended for fixed layout)
- fallback `--discovery_mode walk` for recursive full scan
- default `--skip_incomplete` + `--skip_recent_seconds 120` to avoid in-progress samples

Then launch training with:

```bash
DATASET_NAME_TRACK="/data/shared-vilab/datasets/OpenVid-1M/out_preprocess_openvid_cotracker_preshard_20260402_13490" \
DATASET_META_NAME_TRACK="datasets/internal_datasets/metadata_track.json" \
bash scripts/wan2.1_fun_track/train_track.sh
```

Input mode:
- `INPUT_MODE_TRACK=video` (default): decode video and encode by VAE in training loop
- `INPUT_MODE_TRACK=latent`: load `vae_latents.pt` directly from metadata (`latent_file_path` or `file_path`)
- latent mode automatically uses precomputed `text_feature_wan_t5.npz` (`prompt_embeds`, `attention_mask`) when present
- latent mode automatically uses precomputed `first_frame_clip_feature.npz` as `clip_fea` when present
- latent mode can use precomputed `first_frame_vae_latent.pt` (`first_frame_vae_latent_file_path`) to skip per-step first-frame VAE encoding

Precompute first-frame VAE latents:

```bash
python scripts/wan2.1_fun_track/precompute_first_frame_vae_latent.py \
  --metadata_path "datasets/internal_datasets/metadata_track.json" \
  --data_root "/data/shared-vilab/datasets/OpenVid-1M" \
  --model_name "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"
```

Dummy dataloader in training loop:

```bash
accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun_track/train_track.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP" \
  --input_mode_track latent \
  --dummy_data_track \
  --dummy_length_track 128 \
  --train_batch_size 2 \
  --max_train_steps 2 \
  --use_track_condition
```
