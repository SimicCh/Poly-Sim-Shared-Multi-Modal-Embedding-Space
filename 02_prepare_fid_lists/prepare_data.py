import os


part = 'v3'
language = 'German'

inp_train_fn = f'../../00_data/11_MavCeleb/splits/{part}_train_{language}.csv'
inp_val_fn =   f'../../00_data/11_MavCeleb/splits/{part}_val_{language}.csv'
inp_test_fn =  f'../../00_data/11_MavCeleb/splits/{part}_test_{language}.csv'

out_train_faces_fn = f'./fid_lists/{part}_train_{language}_faces.csv'
out_val_faces_fn = f'./fid_lists/{part}_val_{language}_faces.csv'
out_test_faces_fn = f'./fid_lists/{part}_test_{language}_faces.csv'

out_train_voices_fn = f'./fid_lists/{part}_train_{language}_voices.csv'
out_val_voices_fn = f'./fid_lists/{part}_val_{language}_voices.csv'
out_test_voices_fn = f'./fid_lists/{part}_test_{language}_voices.csv'

out_val_trials_fn = f'./fid_lists/{part}_val_{language}_trials.csv'
out_test_trials_fn = f'./fid_lists/{part}_test_{language}_trials.csv'

with open(inp_train_fn, 'r') as f:
    lines_train = f.readlines()[1:]
    lines_train = [line.strip() for line in lines_train]

with open(inp_val_fn, 'r') as f:
    lines_val = f.readlines()[1:]
    lines_val = [line.strip() for line in lines_val]

with open(inp_test_fn, 'r') as f:
    lines_test = f.readlines()[1:]
    lines_test = [line.strip() for line in lines_test]

lines_train_voices = [l.split(',')[0][2:] for l in lines_train]
lines_train_faces = [l.split(',')[1][2:] for l in lines_train]

lines_val_voices = [l.split(',')[0][2:] for l in lines_val]
lines_val_faces = [l.split(',')[1][2:] for l in lines_val]

lines_test_voices = [l.split(',')[0][2:] for l in lines_test]
lines_test_faces = [l.split(',')[1][2:] for l in lines_test]

csv_lines_train_voices = []
for l in lines_train_voices:
    id_ = l.replace('.wav', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_train_voices.append(f'{id_},{l},{speaker_id}\n')
csv_lines_train_voices = list(set(csv_lines_train_voices)) # remove duplicates

csv_lines_train_faces = []
for l in lines_train_faces:
    id_ = l.replace('.jpg', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_train_faces.append(f'{id_},{l},{speaker_id}\n')
csv_lines_train_faces = list(set(csv_lines_train_faces)) # remove duplicates

# ID,wav,speaker
os.makedirs('./fid_lists/', exist_ok=True)
with open(out_train_voices_fn, 'w') as f:
    f.write('ID,wav,speaker\n')
    for line in csv_lines_train_voices:
        f.write(line)

# ID,img,speaker
with open(out_train_faces_fn, 'w') as f:
    f.write('ID,img,speaker\n')
    for line in csv_lines_train_faces:
        f.write(line)


csv_lines_val_voices = []
for l in lines_val_voices:
    id_ = l.replace('.wav', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_val_voices.append(f'{id_},{l},{speaker_id}\n')
csv_lines_val_voices = list(set(csv_lines_val_voices)) # remove duplicates

csv_lines_val_faces = []
for l in lines_val_faces:
    id_ = l.replace('.jpg', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_val_faces.append(f'{id_},{l},{speaker_id}\n')
csv_lines_val_faces = list(set(csv_lines_val_faces)) # remove duplicates

# ID,wav,speaker
with open(out_val_voices_fn, 'w') as f:
    f.write('ID,wav,speaker\n')
    for line in csv_lines_val_voices:
        f.write(line)

# ID,img,speaker
with open(out_val_faces_fn, 'w') as f:
    f.write('ID,img,speaker\n')
    for line in csv_lines_val_faces:
        f.write(line)


csv_lines_test_voices = []
for l in lines_test_voices:
    id_ = l.replace('.wav', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_test_voices.append(f'{id_},{l},{speaker_id}\n')
csv_lines_test_voices = list(set(csv_lines_test_voices)) # remove duplicates

csv_lines_test_faces = []
for l in lines_test_faces:
    id_ = l.replace('.jpg', '').replace('/', '_')
    speaker_id = l.split('/')[2]
    csv_lines_test_faces.append(f'{id_},{l},{speaker_id}\n')
csv_lines_test_faces = list(set(csv_lines_test_faces)) # remove duplicates

# ID,wav,speaker
with open(out_test_voices_fn, 'w') as f:
    f.write('ID,wav,speaker\n')
    for line in csv_lines_test_voices:
        f.write(line)

# ID,img,speaker
with open(out_test_faces_fn, 'w') as f:
    f.write('ID,img,speaker\n')
    for line in csv_lines_test_faces:
        f.write(line)

# Prepare trials files for val and test sets
val_trials_csv_lines = []
for l in lines_val:
    voice_path = l.split(',')[0][2:]
    face_path = l.split(',')[1][2:]
    label = voice_path.split('/')[2]
    voice_id = voice_path.replace('.wav', '').replace('/', '_')
    face_id = face_path.replace('.jpg', '').replace('/', '_')
    val_trials_csv_lines.append(f'{label},{voice_id},{face_id}\n')
val_trials_csv_lines = list(set(val_trials_csv_lines)) # remove duplicates

with open(out_val_trials_fn, 'w') as f:
    f.write('label,voice_id,face_id\n')
    for line in val_trials_csv_lines:
        f.write(line)


test_trials_csv_lines = []
for l in lines_test:
    voice_path = l.split(',')[0][2:]
    face_path = l.split(',')[1][2:]
    label = voice_path.split('/')[2]
    voice_id = voice_path.replace('.wav', '').replace('/', '_')
    face_id = face_path.replace('.jpg', '').replace('/', '_')
    test_trials_csv_lines.append(f'{label},{voice_id},{face_id}\n')
test_trials_csv_lines = list(set(test_trials_csv_lines)) # remove duplicates

with open(out_test_trials_fn, 'w') as f:
    f.write('label,voice_id,face_id\n')
    for line in test_trials_csv_lines:
        f.write(line)


print('Done.')
