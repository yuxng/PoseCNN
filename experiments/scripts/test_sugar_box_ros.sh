#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_images.py --gpu 0 \
  --network vgg16_convs \
  --model output_napoli/lov/lov_004_sugar_box_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_sugar_box_iter_120000.ckpt \
  --imdb lov_single_004_sugar_box_train \
  --cfg experiments/cfgs/lov_color_sugar_box.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
