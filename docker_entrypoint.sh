#!/bin/bash

cd /workspace/job

git config --global user.email "user@email.com"
git config --global user.name "name"
git remote set-url origin https://github.com/DMLSG/human-protein-atlas.git
git pull
git checkout $BRANCH
git pull

./$TRAIN_SCRIPT
