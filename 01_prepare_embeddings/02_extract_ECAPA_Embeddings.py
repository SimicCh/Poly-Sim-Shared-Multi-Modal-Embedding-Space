import os
import torch
import torchaudio
from tqdm import tqdm
import pickle as pkl

from speechbrain.pretrained import EncoderClassifier
from ECAPA_TDNN.ECAPA_TDNN_mod import ECAPA_TDNN




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/01_prep_file_lists/")
    parser.add_argument("--fids_list", type=str, default="voices_fids.txt")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--output_fids_fn", type=str, default="processed_fids.txt")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/spkrec-ecapa-voxceleb/")
    parser.add_argument("--embedding_layer", type=str, default="prefc")  # options: "final" or "prefc"
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    fids_list = args.fids_list
    output_dir = args.output_dir
    output_fids_fn = args.output_fids_fn
    model_path = args.model_path
    embedding_layer = args.embedding_layer
    nshards = args.nshards
    shard = args.shard

    os.makedirs(output_dir, exist_ok=True)

    print('='*50, flush=True)
    print('Starting ECAPA-TDNN embedding extraction ...', flush=True)
    print('Arguments:', flush=True)
    print('data_dir:', data_dir, flush=True)
    print('fids_list:', fids_list, flush=True)
    print('output_dir:', output_dir, flush=True)
    print('output_fids_fn:', output_fids_fn, flush=True)
    print('model_path:', model_path, flush=True)
    print('embedding_layer:', embedding_layer, flush=True)
    print('nshards:', nshards, flush=True)
    print('shard:', shard, flush=True)
    print('='*50, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)


    # Lade pretrained ECAPA-TDNN Modell (trainiert auf VoxCeleb2)
    classifier = EncoderClassifier.from_hparams(
        source=model_path,
        savedir=model_path,
        overrides={
                "pretrained_path": model_path  # 👈 überschreibt YAML-Eintrag
            }
    )
    classifier.eval()
    classifier.to(device)

    if embedding_layer == 'prefc':
        classifier.mods.embedding_model.forward = classifier.mods.embedding_model.forward_prefc
    elif embedding_layer == 'final':
        pass
    else:
        raise ValueError(f'Invalid embedding_layer: {embedding_layer}. Choose "final" or "prefc".')
    
    # Print number of parameters
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f'Number of classifier parameters: {num_params}', flush=True)
    
    # Load file ids
    with open(fids_list, 'r') as f:
        fids = [line.strip() for line in f.readlines()]

    # Shard fids
    if nshards > 1:
        start_idx = shard * (len(fids) // nshards)
        if shard == nshards - 1:
            end_idx = len(fids)
        else:
            end_idx = (shard + 1) * (len(fids) // nshards)
        fids = fids[start_idx:end_idx]
        print(f'Processing shard {shard+1}/{nshards}: Processing samples {start_idx} to {end_idx} (total {len(fids)}) ...', flush=True)
    else: 
        print(f'Processing {len(fids)} audio samples ...', flush=True)
    
    final_fids = []
    for fid in tqdm(fids, desc="Processing face images"):
        f_path = os.path.join(data_dir, fid)

        # Check if file exists
        if not os.path.isfile(f_path):
            print(f'File not found: {f_path}. Skipping ...', flush=True)
            continue

        out_fn = os.path.join(output_dir, fid.replace('.wav', '.pkl'))
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        
        # load wav file
        wav, fs = torchaudio.load(f_path)  # [channels, time]
        # Das Modell erwartet 16 kHz Samplingrate
        if fs != 16000:
            wav = torchaudio.functional.resample(wav, fs, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = classifier.encode_batch(wav)  # shape: [1,1, emb_dim]
            emb = emb.squeeze(0).squeeze(0).to('cpu').detach()  # shape: [emb_dim]
        
        # save embedding
        with open(out_fn, 'wb') as f:
            pkl.dump(emb, f)
        
        # final_fids.append(fid.replace('.wav', '.pkl'))
        final_fids.append(os.path.relpath(out_fn, output_dir))

    with open(os.path.join(output_dir, output_fids_fn), 'w') as f:
        for fid in final_fids:
            f.write(f'{fid}\n')

    print('Done.')
