#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_2d_pose.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -f $PWD/output/lov/lov_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_80000/segmentations.pkl ]
then
  rm $PWD/output/lov/lov_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_80000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/lov/lov_011_banana_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_banana_iter_160000.ckpt \
  --imdb lov_single_011_banana_train \
  --cfg experiments/cfgs/lov_color_banana.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
