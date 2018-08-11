#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_ros_bag.py --gpu 0 \
  --bag /home/satco/catkin_ws/src/thesis/bag/outside_sun_multiple.bag \
  --network vgg16_convs \
  --model data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt \
  --imdb lov_keyframe \
  --cfg experiments/cfgs/lov_color_2d.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
