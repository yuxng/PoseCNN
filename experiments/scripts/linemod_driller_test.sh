#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_driller_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


if [ -f $PWD/output/linemod/linemod_driller_test/vgg16_fcn_color_single_frame_pose_linemod_driller_iter_80000/segmentations.pkl ]
then
  rm $PWD/output/linemod/linemod_driller_test/vgg16_fcn_color_single_frame_pose_linemod_driller_iter_80000/segmentations.pkl
fi

# test for pose regression
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output_neil/linemod/linemod_driller_train/vgg16_fcn_color_single_frame_pose_linemod_driller_iter_80000.ckpt \
  --imdb linemod_driller_test \
  --cfg experiments/cfgs/linemod_driller_pose.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --background data/cache/backgrounds.pkl
