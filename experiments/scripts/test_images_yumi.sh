#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu 0 \
  --network vgg16_convs \
  --model output/yumi/yumi_train/vgg16_fcn_color_single_frame_2d_pose_add_yumi_iter_1000.ckpt \
  --imdb yumi_train \
  --cfg experiments/cfgs/yumi_color_2d.yml \
  --rig data/yumi/camera.json \
  --cad data/yumi/models.txt \
  --pose data/yumi/poses.txt \
  --background data/cache/backgrounds.pkl
