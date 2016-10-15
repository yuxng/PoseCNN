#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/shapenet_scene_vgg16_depth.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --network vgg16 \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb shapenet_scene_train \
  --cfg experiments/cfgs/shapenet_scene.yml \
  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --network vgg16 \
  --model output/shapenet_scene/shapenet_scene_train/vgg16_fcn_depth_shapenet_scene_iter_40000.ckpt \
  --imdb shapenet_scene_val \
  --cfg experiments/cfgs/shapenet_scene.yml
