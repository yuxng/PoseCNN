#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu 0 \
  --network vgg16_convs \
  --model output/lov/lov_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_40000.ckpt \
  --imdb lov_keyframe \
  --cfg experiments/cfgs/lov_color_2d_pose.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/lov_train_backgrounds.pkl
