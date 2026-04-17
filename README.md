# Poly-Sim Shared Multi-Modal Embedding Space

PyTorch-based research code for learning a **shared multi-modal embedding space** using **Poly-Similarity metric learning** for cross-modal matching tasks (e.g. face–voice association).

This repository contains a **shared multi-modal embedding model trained on MavCeleb** and evaluated using Poly-Similarity.


---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/SimicCh/Poly-Sim-Shared-Multi-Modal-Embedding-Space.git
cd Poly-Sim-Shared-Multi-Modal-Embedding-Space
```

### 2. Prepare venvs
Virtual environment for data preparation:
```bash
sh prep_venv_preparation.sh
```
Virtual environment for SpeechBrain-based training:
```bash
sh prep_venv_preparation.sh
```

### 3. Data preparation
Execute all bash files from [01_prepare_embeddings](./01_prepare_embeddings).
Paths to the MavCeleb or VoxCeleb2 data directories must be adapted accordingly.

### 4. Prepare fid csv files (manifest) for training
The manifest for MavCeleb is provided in [./02_prepare_fid_lists/fid_lists](./02_prepare_fid_lists/fid_lists). The manifest for VoxCeleb2 [./03_prepare_fid_lists_VoxCeleb2](./03_prepare_fid_lists_VoxCeleb2) must be executed if running pre-training is required.

### 5. Pre-Training
Pre-training on VoxCeleb2 (e.g. no-English setup for the English-unheard scenario):
```bash
python3 train_av_simpleProj_speaker_rec_prertrain_VoxCeleb2.py \
    ./hparams_simpleProjection_after_VoxCeleb2_pretraining/train_simpleProjection_pretraining_VoxCeleb2_no_en.yaml
```

### 6. Fine-Tuning
Fine-tuning on MavCeleb corresponding to the pre-training setup (e.g. V3 German):
```bash
python3 train_av_simpleProj_speaker_rec.py \
    ./hparams_simpleProjection_after_VoxCeleb2_pretraining/train_simpleProjection_finetuneMavCelebV3_ge.yaml
```




