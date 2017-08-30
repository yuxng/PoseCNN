#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/test_kinect_fusion.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test icp
time ./tools/test_kinect_fusion.py --gpu 0 \
  --imdb lov_trainval \
  --rig data/LOV/camera.json
