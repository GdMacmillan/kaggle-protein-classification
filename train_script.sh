#!/bin/bash

./setup_cloud_storage.sh

echo "*** running train script ***"

python -u train.py -n "vgg16" -d "hpa" -p "True" --train-images-path="/hpakf-image-data/data/train_images" --nEpochs=1 --batchSz=32 --use-cuda=yes

echo "*** training complete. running test script ***"

GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs_credentials.json python -u test.py -n "vgg16" -d "hpa" -p "True" --batchSz=32 --test-images-path="/hpakf-image-data/data/test_images"

echo "*** testing complete. pod will stop in 10 minutes ***"

sleep 600
