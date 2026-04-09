# Next agent: Wan2.2 Fun I2V / inpaint (VideoX-Fun)

Short handoff. **Architecture changes are out of scope here** (handled by the user). This doc points to the code paths to extend or wire.

## Goal

Train and sanity-check **Wan 2.2 Fun image-to-video** flows in-repo, using the same stack as upstream (Accelerate, `FlowMatchEulerDiscreteScheduler`, `Wan2_2Transformer3DModel`).

## Primary references (read in this order)

| File | Role |
|------|------|
| `scripts/wan2.2_fun/train.sh` | Example launch: `config/wan2.2/wan_civitai_i2v.yaml`, `--train_mode=inpaint`, dataset env vars, resolution / frames / bucket flags. |
| `scripts/wan2.2_fun/train.py` | Full training loop: loads `OmegaConf` from `--config_path`, builds `Wan2_2Transformer3DModel`, `train_mode != "normal"` → I2V / inpaint data path, `transformer3d(..., y=..., clip_fea=...)`. |
| `examples/wan2.2_fun/predict_i2v.py` | Inference script: `Wan2_2FunInpaintPipeline`, schedulers, latent helpers (`get_image_to_video_latent`), multi-GPU / offload knobs at top of file. |
| `config/wan2.2/wan_civitai_i2v.yaml` | Scheduler, VAE, text encoder, transformer kwargs used with the scripts above. |

## Modeling (user-owned)

_(Leave empty for the user’s notes: new blocks, config keys, `forward` signature changes, etc.)_

## Integration checklist (when the user changes the transformer)

- Register / export the class in `videox_fun/models/__init__.py` if it is new.
- `train.py`: `from_pretrained` / validation pipeline must use the same class and `forward` contract (`x`, `t`, `context`, `seq_len`, and for I2V `y`, `clip_fea` as today).
- Keep **`scheduler_kwargs`** in the chosen YAML aligned with any downstream finetune (e.g. `shift`, `num_train_timesteps`, `use_dynamic_shifting`).
- After training, confirm **`predict_i2v.py`** still loads the saved folder (or document a new example script).

## Repo / env

- Package root: `videox_fun/` (install with `pip install -e .` from repo root if needed).
- Training is typically `accelerate launch ... scripts/wan2.2_fun/train.py` (see `train.sh`).
