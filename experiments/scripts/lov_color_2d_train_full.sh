#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_2d_train_full.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

time ./tools/train_net.py --gpu 0 \
  --network vgg16_full \
  --weights data/imagenet_models/vgg16.npy \
  --imdb lov_trainval \
  --cfg experiments/cfgs/lov_color_2d_full.yml \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --iters 120000
