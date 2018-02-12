#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_2d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb lov_trainval \
  --cfg experiments/cfgs/lov_color_2d.yml \
  --iters 80000

time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16.npy \
  --ckpt output/lov/lov_trainval/vgg16_fcn_color_single_frame_2d_lov_iter_80000.ckpt \
  --imdb lov_trainval \
  --cfg experiments/cfgs/lov_color_2d_pose.yml \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --iters 80000
