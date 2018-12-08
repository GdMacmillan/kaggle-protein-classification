#!/bin/bash

python -u train.py -n "vgg16" -d "hpa" -p "True" --train-images-path="/hpakf-image-data/data/train_images" --test-images-path="/hpakf-image-data/data/test_images" --nEpochs=1 --batchSz=32   --use-cuda=yes

echo "*** training complete. pod will stop in 10 minutes ***"

sleep 600
