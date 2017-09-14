#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_can_test_3d.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/linemod/linemod_can_train/vgg16_fcn_color_single_frame_3d_linemod_can_iter_80000.ckpt \
  --imdb linemod_can_test \
  --cfg experiments/cfgs/linemod_can_3d.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --background data/cache/backgrounds.pkl
