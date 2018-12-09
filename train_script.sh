#!/bin/bash

echo "*** running train script ***"

python -u train.py -n "vgg16" -d "hpa" -p "True" --train-images-path="/hpakf-image-data/data/train_images" --nEpochs=1 --batchSz=32 --use-cuda=yes

echo "*** training complete. running test script ***"

python -u test.py -n "vgg16" -d "hpa" -p "True" --test-images-path="/hpakf-image-data/data/test_images"

echo "*** testing complete. pod will stop in 10 minutes ***"

sleep 7200
