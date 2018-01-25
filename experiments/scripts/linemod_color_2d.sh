#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_color_2d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

# train for labeling
#time ./tools/train_net.py --gpu 0 \
#  --network vgg16_convs \
#  --weights data/imagenet_models/vgg16_convs.npy \
#  --imdb linemod_train \
#  --cfg experiments/cfgs/linemod_color_2d.yml \
#  --iters 80000

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/linemod/linemod_train/vgg16_fcn_color_single_frame_2d_linemod_iter_6000.ckpt \
  --imdb linemod_test \
  --cfg experiments/cfgs/linemod_color_2d.yml \
  --background data/cache/backgrounds.pkl
