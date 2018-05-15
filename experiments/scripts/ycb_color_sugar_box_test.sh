#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_color_2d_pose.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output_adrian/ycb/ycb_004_sugar_box_train/vgg16_fcn_color_single_frame_2d_pose_add_ycb_sugar_box_iter_160000.ckpt \
  --imdb ycb_single_004_sugar_box_train \
  --cfg experiments/cfgs/ycb_color_sugar_box.yml \
  --background data/cache/backgrounds.pkl
