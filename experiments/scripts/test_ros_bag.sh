#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_ros_bag.py --gpu 0 \
  --bag data/ROS/dart_2018-05-25-11-49-31.bag \
  --network vgg16_convs \
  --model output/lov/lov_011_banana_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_banana_iter_160000.ckpt \
  --imdb lov_single_011_banana_train \
  --cfg experiments/cfgs/lov_color_banana.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
