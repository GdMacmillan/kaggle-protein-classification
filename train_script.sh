#!/bin/bash

./setup_cloud_storage.sh

echo "*** running train script ***"

python -u train.py -n "resnet152" -d "hpa" -p "True" --train-images-path="/hpakf-image-data/data/train_images" --nEpochs=25 --batchSz=64 --unfreeze-epoch=23 --use-cuda=yes --crit="crl"

echo "*** training complete. running test script ***"

python -u test.py -n "resnet152" -d "hpa" -p "True" --batchSz=64 --test-images-path="/hpakf-image-data/data/test_images"

echo "*** testing complete. pod will stop in 10 minutes ***"

sleep 600
