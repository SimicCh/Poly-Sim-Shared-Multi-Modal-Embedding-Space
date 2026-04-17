#!/bin/bash -l
module load python/python-3.13.0
python -m venv venv_sb_
source venv_sb_/bin/activate
pip install --upgrade pip
pip install -r requirements_sb.txt
echo 'Done.'
