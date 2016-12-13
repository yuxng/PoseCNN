#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
# export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/gmu_scene_vgg16_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for multiple frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16 \
  --model output/gmu_scene/gmu_scene_train/vgg16_fcn_rgbd_multi_frame_gmu_scene_iter_15000.ckpt \
  --imdb gmu_scene_train \
  --cfg experiments/cfgs/gmu_scene.yml \
  --rig lib/kinect_fusion/data/camera.json
