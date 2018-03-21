#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_color_2d_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -f $PWD/output/ycb/ycb_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_ycb_iter_80000/segmentations.pkl ]
then
  rm $PWD/output/ycb/ycb_keyframe/vgg16_fcn_color_single_frame_2d_pose_add_ycb_iter_80000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/ycb/ycb_trainval/vgg16_fcn_color_single_frame_2d_pose_add_ycb_iter_80000.ckpt \
  --imdb ycb_trainval \
  --cfg experiments/cfgs/ycb_color_2d_pose.yml \
  --rig data/ycb/camera.json \
  --cad data/ycb/models.txt \
  --pose data/ycb/poses.txt \
  --background data/cache/backgrounds.pkl
