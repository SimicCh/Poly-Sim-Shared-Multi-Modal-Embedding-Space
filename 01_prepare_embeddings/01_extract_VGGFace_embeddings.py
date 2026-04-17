
import torch
import torchvision.io as io
from torchvision import transforms
import torchfile
from tqdm import tqdm
import pickle as pkl
import cv2
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VGG_Face_Model.vgg_face_pytorch.models.vgg_face import VGG_16

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
    parser.add_argument("--model_path", type=str, default="./pretrained_models/spkrec-ecapa-voxceleb/")
    parser.add_argument("--embedding_layer", type=str, default="fc6")  # options: 'fc6', 'fc7', 'fc8'
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)

    args = parser.parse_args()

    data_dir = args.data_dir
    fids_list = args.fids_list
    output_dir = args.output_dir
    output_fids_fn = args.output_fids_fn
    model_path = args.model_path
    embedding_layer = args.embedding_layer
    chunk_size = args.chunk_size
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
    print('model_path:', model_path, flush=True)
    print('embedding_layer:', embedding_layer, flush=True)
    print('chunk_size:', chunk_size, flush=True)
    print('nshards:', nshards, flush=True)
    print('shard:', shard, flush=True)

    print('='*50, flush=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print(f'Loading model from: {model_path}', flush=True)
    model = VGG_16() #.double()
    model.load_weights(path=model_path)
    model.to(device)
    model.eval()
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {num_params}', flush=True)
    
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

        # Wenn out_fn existiert, überspringen
        if os.path.isfile(out_fn):
            print(f"Skip {out_fn}, already exists.", flush=True)
            final_fids.append(os.path.relpath(out_fn, output_dir))
            continue

        img_tensor = io.read_image(f_path)  # load image shape: [3, H, W], dtype: uint8 (RGB)
        img_tensor = img_tensor[[2,1,0],:,:] # RGB to BGR
        img_tensor = transform(img_tensor).unsqueeze(0).float() # shape: [1, 3, 224, 224]
        img_tensor -= torch.tensor([129.1863, 104.7624, 93.5940], dtype=torch.float).view(1, 3, 1, 1)

        with torch.no_grad():
            embeddings = model.get_embedding(img_tensor.to(device), layer=embedding_layer).to('cpu').detach()  # embedding shape: [emb_dim]
            embeddings = embeddings.squeeze()
            
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        with open(out_fn, 'wb') as f:
            pkl.dump(embeddings, f)

        final_fids.append(os.path.relpath(out_fn, output_dir))

    with open(os.path.join(output_dir, output_fids_fn), 'w') as f:
        for fid in final_fids:
            f.write(f'{fid}\n')


    print('Done.')
