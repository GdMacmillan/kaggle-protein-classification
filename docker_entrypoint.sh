#!/bin/bash

cd /workspace/job

python -u train.py -n "basicnet01" -d "hpa" -no-cuda=False --train-images-path="/hpakf-image-data/data/train_images"
sleep 100000
