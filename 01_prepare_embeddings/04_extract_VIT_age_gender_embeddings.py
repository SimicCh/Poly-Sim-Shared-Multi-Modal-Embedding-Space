
import torch
import torchvision.io as io
from torchvision import transforms
import torchfile
from tqdm import tqdm
import pickle as pkl
import cv2
import random

import sys, os

from huggingface_hub import snapshot_download

sys.path.append("./abhilash88/age-gender-prediction")

from transformers import AutoFeatureExtractor, AutoModel, AutoModelForImageClassification
from transformers import pipeline
from model import predict_age_gender

from transformers import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers import AutoConfig, AutoImageProcessor

from model import AgeGenderViTModel

# import abhilash88.age_gender_prediction.model as age_gender_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects this size
    # transforms.ConvertImageDtype(torch.float32),
    # transforms.ToTensor(),
])

random.seed(42)
torch.manual_seed(42)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/01_prep_file_lists/")
    parser.add_argument("--fids_list", type=str, default="voices_fids.txt")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--output_fids_fn", type=str, default="processed_fids.txt")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="abhilash88/age-gender-prediction")
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)

    args = parser.parse_args()

    data_dir = args.data_dir
    fids_list = args.fids_list
    output_dir = args.output_dir
    output_fids_fn = args.output_fids_fn
    step_size = args.step_size
    model_name = args.model_name
    nshards = args.nshards
    shard = args.shard

    os.makedirs(output_dir, exist_ok=True)

    # Logging
    print('='*50, flush=True)
    print('Starting VGG Face embedding extraction ...', flush=True)
    print('Arguments:', flush=True)
    print('data_dir:', data_dir, flush=True)
    print('fids_list:', fids_list, flush=True)
    print('output_dir:', output_dir, flush=True)
    print('output_fids_fn:', output_fids_fn, flush=True)
    print('step_size:', step_size, flush=True)
    print('model_name:', model_name, flush=True)
    print('nshards:', nshards, flush=True)
    print('shard:', shard, flush=True)
    print('='*50, flush=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print(f'Loading model: {model_name}', flush=True)
    # Load model using the custom class
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AgeGenderViTModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True
    )
    model.eval()
    model.to(device)
    
    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    # Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    # # Print number of parameters
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f'Number of model parameters: {num_params}', flush=True)
    
    with open(fids_list, 'r') as f:
        fids = [line.strip() for line in f.readlines()]
    
    # Sharding
    if nshards > 1:
        start_idx = shard * (len(fids) // nshards)
        end_idx = (shard + 1) * (len(fids) // nshards)
    else:
        start_idx = 0
        end_idx = len(fids)
    
    fids = fids[start_idx:end_idx]
    print(f'Processing {len(fids)} face image folders from index {start_idx} to {end_idx}', flush=True)


    final_fids = []
    for fid in tqdm(fids, desc="Processing face images"):
        f_path = os.path.join(data_dir, fid)
        out_fn = os.path.join(output_dir, fid.replace('.jpg', '.pkl'))
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)

        # Wenn out_fn existiert, überspringen
        if os.path.isfile(out_fn):
            print(f"Skip {out_fn}, already exists.", flush=True)
            final_fids.append(os.path.relpath(out_fn, output_dir))
            continue

        img_tensor = io.read_image(f_path)
        img_tensor = transform(img_tensor).unsqueeze(0).float() # shape: [1, 3, 224, 224]
        inputs = processor(images=img_tensor, return_tensors="pt", do_center_crop=True, crop_size=224)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.get_embedding(inputs['pixel_values'].to(device)).to('cpu').detach()  # embedding shape: [emb_dim]
            embeddings = embeddings.squeeze()

        with open(out_fn, 'wb') as f:
            pkl.dump(embeddings, f)

        final_fids.append(os.path.relpath(out_fn, output_dir))


    with open(os.path.join(output_dir, output_fids_fn), 'w') as f:
        for fid in final_fids:
            f.write(f'{fid}\n')


    print('Done.')
