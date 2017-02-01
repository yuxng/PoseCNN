#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/rgbd_scene_single_depth.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb rgbd_scene_train \
  --cfg experiments/cfgs/rgbd_scene_single_depth.yml \
  --iters 40000

if [ -f $PWD/output/rgbd_scene/rgbd_scene_val/vgg16_fcn_depth_single_frame_rgbd_scene_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/rgbd_scene/rgbd_scene_val/vgg16_fcn_depth_single_frame_rgbd_scene_iter_40000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/rgbd_scene/rgbd_scene_train/vgg16_fcn_depth_single_frame_rgbd_scene_iter_40000.ckpt \
  --imdb rgbd_scene_val \
  --cfg experiments/cfgs/rgbd_scene_single_depth.yml \
  --rig data/RGBDScene/camera.json
