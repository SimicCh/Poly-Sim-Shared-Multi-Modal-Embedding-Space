# speaker_embeddings_wav2vec2.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import pickle as pkl

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class ModelHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
        ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender




def process_func(
    wav_path,
    model,
    processor,
    embeddings: bool = False,
    ) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    audio, sr = torchaudio.load(wav_path)  # [channels, time]
    # audio, sr = sf.read(wav_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
        sr = 16000
    
    if len(audio.shape) > 1 and  audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze() # [time]

    y = processor(audio, sampling_rate=sr)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0].squeeze(0)
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = torch.tensor(y).detach().cpu() #.detach().cpu().numpy()

    return y



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/01_prep_file_lists/")
    parser.add_argument("--fids_list", type=str, default="voices_fids.txt")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--output_fids_fn", type=str, default="processed_fids.txt")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    fids_list = args.fids_list
    output_dir = args.output_dir
    output_fids_fn = args.output_fids_fn
    model_name = args.model_name
    nshards = args.nshards
    shard = args.shard
    os.makedirs(output_dir, exist_ok=True)

    print('='*50, flush=True)
    print('Starting wav2vec2 embedding extraction ...', flush=True)
    print('Arguments:', flush=True)
    print('data_dir:', data_dir, flush=True)
    print('fids_list:', fids_list, flush=True)
    print('output_dir:', output_dir, flush=True)
    print('output_fids_fn:', output_fids_fn, flush=True)
    print('model_name:', model_name, flush=True)
    print('nshards:', nshards, flush=True)
    print('shard:', shard, flush=True)
    print('='*50, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    # load model from hub
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AgeGenderModel.from_pretrained(model_name)
    model.eval()
    model.to(device)


    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {num_params}', flush=True)
    
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
        print(f'Processing shard {shard+1}/{nshards}: indices {start_idx} to {end_idx} ...', flush=True)
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
        if os.path.isfile(out_fn):
            print(f'Embedding already exists: {out_fn}. Skipping ...', flush=True)
            final_fids.append(os.path.relpath(out_fn, output_dir))
            continue
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        
        # Extrahiere wav2vec2 Embedding
        embedding = process_func(
                        wav_path=f_path,
                        model=model,
                        processor=processor,
                        embeddings=True
                )

        # Speichere das Embedding
        with open(out_fn, 'wb') as f:
            pkl.dump(embedding, f)

        # final_fids.append(fid.replace('.wav', '.pkl'))
        final_fids.append(os.path.relpath(out_fn, output_dir))

    with open(os.path.join(output_dir, output_fids_fn), 'w') as f:
        for fid in final_fids:
            f.write(f'{fid}\n')

print('Done.')
