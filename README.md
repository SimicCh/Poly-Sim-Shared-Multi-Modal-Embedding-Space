# Poly-Sim Shared Multi-Modal Embedding Space

PyTorch-based research code for learning a **shared multi-modal embedding space** using **Poly-Similarity metric learning** for cross-modal matching tasks (e.g. face–voice association).

PyTorch-based research code for a **shared multi-modal embedding space** model trained on MavCeleb for Poly-Sim evaluation.

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/SimicCh/Poly-Sim-Shared-Multi-Modal-Embedding-Space.git
cd Poly-Sim-Shared-Multi-Modal-Embedding-Space
```

### 2. Prepare venvs
Virtual environment for data preparation.
```bash
sh prep_venv_preparation.sh
```
Virtual environment for speechbrain based training.
```bash
sh prep_venv_preparation.sh
```

### 3. Data preparation
Execute all bash files from [01_prepare_embeddings](./01_prepare_embeddings).
Therefore paths to MavCeleb or VoxCeleb2 data directories must be changed.

### 4. Pre-Training
Pre-training on VoxCeleb2 (e.g. for no englsh setting, which will be used as english unheard scenario)
```bash
python3 train_av_simpleProj_speaker_rec_prertrain_VoxCeleb2.py \
    ./hparams_simpleProjection_after_VoxCeleb2_pretraining/train_simpleProjection_pretraining_VoxCeleb2_no_en.yaml
```

### 5. Fine-Tuning
Fine-Tuning on MavCeleb is executed accordingly to pre-training (e.g. V3 german)
```bash
python3 train_av_simpleProj_speaker_rec.py \
    ./hparams_simpleProjection_after_VoxCeleb2_pretraining/train_simpleProjection_finetuneMavCelebV3_ge.yaml
```




