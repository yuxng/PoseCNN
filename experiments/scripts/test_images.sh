#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu 0 \
  --network vgg16_convs \
  --model output/ycb/ycb_trainval/vgg16_fcn_color_single_frame_2d_ycb_iter_120000.ckpt \
  --imdb ycb_trainval \
  --cfg experiments/cfgs/ycb_color_2d.yml \

#time ./tools/test_images.py --gpu 0 \
#  --network vgg16_convs \
#  --model output/lov/lov_trainval/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_80000.ckpt \
#  --imdb lov_keyframe \
#  --cfg experiments/cfgs/lov_color_2d_pose.yml \
#  --rig data/LOV/camera.json \
#  --cad data/LOV/models.txt \
#  --pose data/LOV/poses.txt \
#  --background data/cache/lov_train_backgrounds.pkl
