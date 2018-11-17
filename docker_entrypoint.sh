#!/bin/bash

pip install gsutil
pip install setproctitle
conda install -y pandas
conda install -c conda-forge scikit-image

python -u job/train.py -n "basicnet01" -d "hpa" -m "True" -no-cuda=False
