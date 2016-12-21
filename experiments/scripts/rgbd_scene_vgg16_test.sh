#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
# export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/rgbd_scene_vgg16_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for multiple frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16 \
  --model output/rgbd_scene/rgbd_scene_train/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_35000.ckpt \
  --imdb rgbd_scene_val \
  --cfg experiments/cfgs/rgbd_scene.yml \
  --rig data/RGBDScene/camera.json
