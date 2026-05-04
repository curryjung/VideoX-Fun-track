preprocess_root="/data/shared-vilab/qihoo360/WISA-80K/out_preprocess_wisa_cotracker_full_20260501_054758"

cd /data/project-vilab/jaeseok/VideoX-Fun

python scripts/wan2.1_fun_track/build_metadata_track.py \
  --preprocess_root "${preprocess_root}" \
  --output_meta "datasets/internal_datasets/metadata_track_wisa.json" \
  --data_root "/data/shared-vilab/qihoo360/WISA-80K" \
  --sample_media latent