#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/test_images.py --gpu 0 \
  --network vgg16_convs \
  --model output/ycb/ycb_trainval/vgg16_fcn_color_single_frame_2d_ycb_iter_20000.ckpt \
  --imdb ycb_trainval \
  --cfg experiments/cfgs/ycb_color_2d.yml \

#time ./ros/test_images.py --gpu 0 \
#  --network vgg16_convs \
#  --model output/ycb/ycb_trainval/vgg16_fcn_color_single_frame_2d_pose_add_ycb_iter_120000.ckpt \
#  --imdb ycb_trainval \
#  --cfg experiments/cfgs/ycb_color_2d_pose.yml \
#  --rig data/ycb/camera.json \
#  --cad data/ycb/models.txt \
#  --pose data/ycb/poses.txt \
#  --background data/cache/ycb_train_backgrounds.pkl
