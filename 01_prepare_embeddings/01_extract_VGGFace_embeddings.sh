#!/bin/bash
echo "Current working directory: $(pwd)"
source ../venv_preparation_/bin/activate

# Prepare Embeddings
python3 01_extract_VGGFace_embeddings.py \
        --data_dir PATH_TO_MAVCELEB \
        --fids_list LIST_WITH_ALL_IMG_FIDS \
        --output_dir OUTPUT_DIR/VGGFace_embeddings/fc6 \
        --output_fids_fn processed_fids.txt \
        --model_path ./VGG_Face_Model/vgg_face_pytorch/pretrained/VGG_FACE.t7 \
        --embedding_layer fc6 \

echo 'Done.'
