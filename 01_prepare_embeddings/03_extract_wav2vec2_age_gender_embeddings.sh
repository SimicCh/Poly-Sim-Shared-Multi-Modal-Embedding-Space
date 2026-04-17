#!/bin/bash
echo "Current working directory: $(pwd)"
source ../venv_preparation_/bin/activate

# Prepare Embeddings
python3 03_extract_wav2vec2_age_gender_embeddings.py \
        --data_dir PATH_TO_MAVCELEB \
        --fids_list LIST_WITH_ALL_WAV_FIDS \
        --output_dir OUTPUT_DIR/wav2vec2_large_age_gender_embeddings \
        --output_fids_fn processed_wav2vec2_embeddings_shard_0_1.txt \
        --model_name audeering/wav2vec2-large-robust-24-ft-age-gender \
        --nshards 1 \
        --shard 0

echo 'Done.'
