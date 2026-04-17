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

