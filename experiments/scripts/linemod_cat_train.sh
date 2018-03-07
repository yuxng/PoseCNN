#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_cat_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

# train for pose
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16.npy \
  --imdb linemod_cat_train \
  --cfg experiments/cfgs/linemod_cat_pose.yml \
  --iters 80000
