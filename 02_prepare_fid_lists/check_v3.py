import os


part = 'v3'
language1 = 'English'
language2 = 'German'

inp_l1_train_fn = f'../../00_data/11_MavCeleb/splits/{part}_train_{language1}.csv'
inp_l1_val_fn =   f'../../00_data/11_MavCeleb/splits/{part}_val_{language1}.csv'
inp_l1_test_fn =  f'../../00_data/11_MavCeleb/splits/{part}_test_{language1}.csv'

inp_l2_train_fn = f'../../00_data/11_MavCeleb/splits/{part}_train_{language2}.csv'
inp_l2_val_fn =   f'../../00_data/11_MavCeleb/splits/{part}_val_{language2}.csv'
inp_l2_test_fn =  f'../../00_data/11_MavCeleb/splits/{part}_test_{language2}.csv'


with open(inp_l1_train_fn, 'r') as f:
    lines_l1_train = f.readlines()[1:]
    lines_l1_train = [line.strip() for line in lines_l1_train]

with open(inp_l1_val_fn, 'r') as f:
    lines_l1_val = f.readlines()[1:]
    lines_l1_val = [line.strip() for line in lines_l1_val]

with open(inp_l1_test_fn, 'r') as f:
    lines_l1_test = f.readlines()[1:]
    lines_l1_test = [line.strip() for line in lines_l1_test]

with open(inp_l2_train_fn, 'r') as f:
    lines_l2_train = f.readlines()[1:]
    lines_l2_train = [line.strip() for line in lines_l2_train]

with open(inp_l2_val_fn, 'r') as f:
    lines_l2_val = f.readlines()[1:]
    lines_l2_val = [line.strip() for line in lines_l2_val]

with open(inp_l2_test_fn, 'r') as f:
    lines_l2_test = f.readlines()[1:]
    lines_l2_test = [line.strip() for line in lines_l2_test]


labels_l1_train = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l1_train] + [l.split(',')[1].split('/')[3] for l in lines_l1_train]))
labels_l1_val = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l1_val] + [l.split(',')[1].split('/')[3] for l in lines_l1_val]))
labels_l1_test = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l1_test] + [l.split(',')[1].split('/')[3] for l in lines_l1_test]))

labels_l2_train = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l2_train] + [l.split(',')[1].split('/')[3] for l in lines_l2_train]))
labels_l2_val = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l2_val] + [l.split(',')[1].split('/')[3] for l in lines_l2_val]))
labels_l2_test = sorted(set([l.split(',')[0].split('/')[3] for l in lines_l2_test] + [l.split(',')[1].split('/')[3] for l in lines_l2_test]))

print(f'Number of unique labels in {language1} train: {len(labels_l1_train)}')
print(f'Number of unique labels in {language1} val: {len(labels_l1_val)}')
print(f'Number of unique labels in {language1} test: {len(labels_l1_test)}')
print(f'Number of unique labels in {language2} train: {len(labels_l2_train)}')
print(f'Number of unique labels in {language2} val: {len(labels_l2_val)}')
print(f'Number of unique labels in {language2} test: {len(labels_l2_test)}')


# check if labels are disjoint between train, val, and test for each language
for i in range(len(labels_l1_train)):
    if labels_l1_train[i] != labels_l1_val[i] or labels_l1_train[i] != labels_l1_test[i] or labels_l1_val[i] != labels_l1_test[i]:
        print(f'train {labels_l1_train[i]} | val {labels_l1_val[i]} | test {labels_l1_test[i]}')

for i in range(len(labels_l2_train)):
    if labels_l2_train[i] != labels_l2_val[i] or labels_l2_train[i] != labels_l2_test[i] or labels_l2_val[i] != labels_l2_test[i]:
        print(f'train {labels_l2_train[i]} | val {labels_l2_val[i]} | test {labels_l2_test[i]}')


for i in range(len(labels_l1_train)):
    if labels_l1_train[i] != labels_l2_train[i]:
        print(f'train {labels_l1_train[i]} | train {labels_l2_train[i]}')

for i in range(len(labels_l1_val)):
    print(f'language1 train {labels_l1_val[i]} | language2 train {labels_l2_val[i]}')

print('Done.')
