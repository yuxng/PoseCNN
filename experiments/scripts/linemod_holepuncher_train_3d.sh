#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_holepuncher_train_3d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

# train for labeling
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb linemod_holepuncher_train \
  --cfg experiments/cfgs/linemod_holepuncher_3d.yml \
  --iters 80000
