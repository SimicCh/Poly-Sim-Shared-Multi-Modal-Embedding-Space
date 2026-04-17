#!/bin/bash
echo "Current working directory: $(pwd)"
source ../venv_preparation_/bin/activate

# Prepare Embeddings
python3 02_extract_ECAPA_Embeddings.py \
        --data_dir PATH_TO_MAVCELEB \
        --fids_list LIST_WITH_ALL_WAV_FIDS \
        --output_dir OUTPUT_DIR/ecapa_embeddings/originalECAPA_prefc \
        --output_fids_fn processed_ECAPA_prefc_embeddings_shard_0_1.txt \
        --model_path ./ECAPA_TDNN/pretrained_ecapa \
        --embedding_layer prefc \
        --nshards 1 \
        --shard 0

echo 'Done.'
