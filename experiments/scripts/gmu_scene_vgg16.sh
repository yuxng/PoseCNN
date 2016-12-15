#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/gmu_scene_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for multiple frames
time ./tools/train_net.py --gpu 0 \
  --network vgg16 \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb gmu_scene_train \
  --cfg experiments/cfgs/gmu_scene.yml \
  --iters 40000
