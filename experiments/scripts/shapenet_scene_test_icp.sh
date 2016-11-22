#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/shapenet_scene_test_icp.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test icp
time ./tools/test_icp.py --gpu $1 \
  --imdb shapenet_scene_val \
  --cfg experiments/cfgs/shapenet_scene.yml
