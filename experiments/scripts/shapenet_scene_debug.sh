#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/shapenet_scene_debug.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --network test_net \
  --imdb shapenet_scene_train \
  --cfg experiments/cfgs/shapenet_scene.yml \
  --iters 40000
