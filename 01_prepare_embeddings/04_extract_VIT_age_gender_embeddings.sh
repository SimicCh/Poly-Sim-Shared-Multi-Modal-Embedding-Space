#!/bin/bash
echo "Current working directory: $(pwd)"
source ../venv_preparation_/bin/activate

# Prepare Embeddings
python3 04_extract_VIT_age_gender_embeddings.py \
        --data_dir PATH_TO_MAVCELEB \
        --fids_list LIST_WITH_ALL_IMG_FIDS \
        --output_dir OUTPUT_DIR/VIT_age_gender_embeddings \
        --output_fids_fn processed_fids_shard__v3_0_1.txt \
        --model_name ./abhilash88/age-gender-prediction \
        --nshards 1 \
        --shard 0

echo 'Done.'
