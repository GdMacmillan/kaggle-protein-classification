#!/bin/bash

cd /workspace/job
python -u train.py -n "basicnet01" -d "hpa" -no-cuda=False
