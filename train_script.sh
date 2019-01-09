#!/bin/bash

./setup_cloud_storage.sh

echo "*** running train script ***"

# GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs_credentials.json python -u train.py -n "vgg16" -d "hpa" -p "True" --train-images-path="/hpakf-image-data/data/train_images" --nEpochs=25 --batchSz=32 --use-cuda=yes --crit="crl"

echo "*** training complete. running test script ***"

GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs_credentials.json python -u test.py -n "vgg16" -d "hpa" -p "True" --batchSz=32 --test-images-path="/hpakf-image-data/data/test_images" --thresholds="-0.03460776160188329,-0.46379391075352083,-0.12344726279967405,484652.98112142825,-0.47770335547110016,575432.7272718627,267711.61211275373,867688.9748014846,-423422.9698672719,-230215.21544145286,671945.7782180005,-740992.41389335,431665.4495568584,259355.27379488765,-0.05314672572842815,-1174584.2785543378,-80781.55565148695,1035021.4936208843,3727975.1370291063,-169612.6619126104,-145378.05353325934,-0.20517406441160552,94615.00568209491,-0.43148805695107884,784203.5853427106,-0.05585992273535652,-393935.39490081306,-889874.6976270062"

echo "*** testing complete. pod will stop in 10 minutes ***"

sleep 600
