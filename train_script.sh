#!/bin/bash

git checkout master

python -u train.py -n "basicnet01" -d "hpa" --train-images-path="/hpakf-image-data/data/train_images" --test-images-path="/hpakf-image-data/data/test_images" --nEpochs=1 --batchSz=256  --use-cuda=yes

echo "*** training complete. pod will stop in 10 minutes ***"

sleep 600