#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_ape_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/linemod/linemod_ape_train/vgg16_fcn_rgbd_single_frame_pose_linemod_ape_iter_40000.ckpt \
  --imdb linemod_ape_test \
  --cfg experiments/cfgs/linemod_ape_pose.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --background data/cache/linemod_ape_train_backgrounds.pkl
