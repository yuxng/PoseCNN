#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_color_2d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

#time ./tools/train_net.py --gpu 0 \
#  --network vgg16_convs \
#  --weights data/imagenet_models/vgg16_convs.npy \
#  --imdb ycb_trainval \
#  --cfg experiments/cfgs/ycb_color_2d.yml \
#  --iters 20000

time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16.npy \
  --ckpt output/ycb/ycb_trainval/vgg16_fcn_color_single_frame_2d_ycb_iter_20000.ckpt \
  --imdb ycb_trainval \
  --cfg experiments/cfgs/ycb_color_2d_pose.yml \
  --iters 120000
