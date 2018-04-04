#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_sugar_box_train_adapt.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --ckpt output_napoli/lov/lov_004_sugar_box_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_sugar_box_iter_120000.ckpt \
  --imdb lov_single_004_sugar_box_train \
  --cfg experiments/cfgs/lov_color_sugar_box_adapt.yml \
  --iters 20000
