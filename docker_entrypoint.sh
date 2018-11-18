#!/bin/bash

cd /workspace/job

python -u train.py -n "basicnet01" -d "hpa" -no-cuda=False --train-images-path="/mnt/human-protein-atlas-data/train_images"
