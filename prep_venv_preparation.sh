#!/bin/bash -l
module load python/python-3.13.0
python -m venv venv_preparation_
source venv_preparation_/bin/activate
pip install --upgrade pip
pip install torch torchvision 
pip install torchaudio
pip install torchcodec
pip install -r requirements_preparation.txt
echo 'Done.'
