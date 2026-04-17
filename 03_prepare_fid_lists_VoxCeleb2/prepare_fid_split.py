import random
random.seed(42)


exclude_lang = 'de'

fids_fn = f'filtered_fids.txt'
metadata_fn = 'audio_clips_meta_data.csv'

train_val_ratio = 0.95

out_train_faces_fn = f'train_faces_no_{exclude_lang}.csv'
out_train_voices_fn = f'train_voices_no_{exclude_lang}.csv'
out_val_faces_fn = f'val_faces_no_{exclude_lang}.csv'
out_val_voices_fn = f'val_voices_no_{exclude_lang}.csv'
out_val_trials_fn = f'val_trials_no_{exclude_lang}.csv'



with open(fids_fn, 'r') as f:
    fids = f.read().splitlines()
    fids = [f.strip() for f in fids]

with open(metadata_fn, 'r') as f:
    metadata = f.read().splitlines()
    metadata = [m.strip() for m in metadata[1:]]  # Skip header line

fid_lang_dict = {}
for m in metadata:
    m_split = m.split(',')
    fid = m_split[-1].replace('.m4a', '')
    lang = m_split[1]
    fid_lang_dict[fid] = lang

print(f'Total FIDs: {len(fids)}')
print(f'Total Metadata Entries: {len(metadata)}')

filtered_fids = [fid for fid in fids if fid_lang_dict.get(fid) != exclude_lang]
print(f'Filtered FIDs (excluding language "{exclude_lang}"): {len(filtered_fids)}')

filtered_fids = sorted(filtered_fids)
random.shuffle(filtered_fids)

fids_train = filtered_fids[:int(len(filtered_fids) * train_val_ratio)]
fids_val = filtered_fids[int(len(filtered_fids) * train_val_ratio):]

print(f'Train FIDs: {len(fids_train)}')
print(f'Validation FIDs: {len(fids_val)}')

train_speaker_ids = sorted(set([fid.split('/')[0] for fid in fids_train]))
print(f'Train Speaker IDs: {len(train_speaker_ids)}')

for fid in fids_val:
    speaker_id = fid.split('/')[0]
    if speaker_id not in train_speaker_ids:
        fids_train.append(fid)
        train_speaker_ids.append(speaker_id)
        fids_val.remove(fid)
        print(f'Moved FID {fid} from validation to training set to ensure speaker overlap.')

fids_train = sorted(fids_train)
fids_val = sorted(fids_val)



# Create Trials
speakers_val = sorted(set([fid.split('/')[0] for fid in fids_val]))
speaker_fids_dict = {speaker: [] for speaker in speakers_val}
for fid in fids_val:
    speaker = fid.split('/')[0]
    speaker_fids_dict[speaker].append(fid)

num_trials = 1000
trials_lines = ['label,voice_id,face_id']
trials_val_fids = []
for _ in range(num_trials):
    speaker = random.choice(speakers_val)
    face_fid = random.choice(speaker_fids_dict[speaker]).replace('/', '_')
    voice_fid = random.choice(speaker_fids_dict[speaker]).replace('/', '_')
    trials_val_fids.extend(face_fid)
    trials_val_fids.extend(voice_fid)
    trials_lines.append(f'{speaker},{voice_fid},{face_fid}')  # Positive trial

trials_val_fids = list(set(trials_val_fids))

for fid in fids_val:
    if fid not in trials_val_fids:
        fids_train.append(fid)

fids_train = sorted(fids_train)
fids_val = sorted(fids_val)


lines_train_faces = ['ID,img,speaker']
for fid in fids_train:
    id_ = fid.replace('/', '_')
    img = f'{fid}.jpg'
    speaker = fid.split('/')[0]
    lines_train_faces.append(f'{id_},{img},{speaker}')

lines_train_voices = ['ID,wav,speaker']
for fid in fids_train:
    id_ = fid.replace('/', '_')
    audio = f'{fid}.wav'
    speaker = fid.split('/')[0]
    lines_train_voices.append(f'{id_},{audio},{speaker}')

lines_val_faces = ['ID,img,speaker']
for fid in fids_val:
    id_ = fid.replace('/', '_')
    img = f'{fid}.jpg'
    speaker = fid.split('/')[0]
    lines_val_faces.append(f'{id_},{img},{speaker}')

lines_val_voices = ['ID,wav,speaker']
for fid in fids_val:
    id_ = fid.replace('/', '_')
    audio = f'{fid}.wav'
    speaker = fid.split('/')[0]
    lines_val_voices.append(f'{id_},{audio},{speaker}')


with open(out_train_faces_fn, 'w') as f:
    f.write('\n'.join(lines_train_faces))

with open(out_train_voices_fn, 'w') as f:
    f.write('\n'.join(lines_train_voices))


with open(out_val_faces_fn, 'w') as f:
    f.write('\n'.join(lines_val_faces))

with open(out_val_voices_fn, 'w') as f:
    f.write('\n'.join(lines_val_voices))

with open(out_val_trials_fn, 'w') as f:
    f.write('\n'.join(trials_lines))




print('Done.')
