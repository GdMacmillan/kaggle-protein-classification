#!/bin/bash

cd /workspace/job

git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git remote set-url origin https://github.com/DMLSG/human-protein-atlas.git
git pull

./train_script.sh