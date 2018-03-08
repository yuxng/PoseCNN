#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/yumi_color_2d_pose.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -f $PWD/output/yumi/yumi_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_yumi_iter_80000/segmentations.pkl ]
then
  rm $PWD/output/yumi/yumi_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_yumi_iter_80000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/yumi/yumi_train/vgg16_fcn_color_single_frame_2d_pose_add_yumi_iter_1000.ckpt \
  --imdb yumi_train \
  --cfg experiments/cfgs/yumi_color_2d.yml \
  --rig data/yumi/camera.json \
  --cad data/yumi/models.txt \
  --pose data/yumi/poses.txt \
  --background data/cache/backgrounds_depth.pkl
