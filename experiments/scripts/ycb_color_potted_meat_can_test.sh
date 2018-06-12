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
  --model output/ycb/ycb_010_potted_meat_can_train/vgg16_fcn_color_single_frame_2d_pose_add_ycb_potted_meat_can_iter_160000.ckpt \
  --imdb lov_single_010_potted_meat_can_keyframe \
  --cfg experiments/cfgs/ycb_color_potted_meat_can.yml \
  --background data/cache/backgrounds.pkl
