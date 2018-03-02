#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_2d_pose.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#if [ -f $PWD/output/lov/lov_val/vgg16_fcn_color_single_frame_2d_lov_iter_80000/segmentations.pkl ]
#then
#  rm $PWD/output/lov/lov_val/vgg16_fcn_color_single_frame_2d_lov_iter_80000/segmentations.pkl
#fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_det \
  --model output/lov/lov_trainval/vgg16_fcn_detection_lov_iter_160000.ckpt \
  --imdb lov_keyframe \
  --cfg experiments/cfgs/lov_det.yml \
  --background data/cache/backgrounds.pkl
