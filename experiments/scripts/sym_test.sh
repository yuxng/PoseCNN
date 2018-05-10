#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/sym_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/sym/sym_train/vgg16_fcn_color_single_frame_2d_pose_sym_sym_iter_10000.ckpt \
  --imdb sym_train \
  --cfg experiments/cfgs/sym.yml \
  --cad data/SYM/models.txt \
  --pose data/SYM/poses.txt \
  --background data/cache/backgrounds.pkl
