#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/rgbd_scene_single_color_fcn8.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
time ./tools/train_net.py --gpu 0 \
  --network fcn8_vgg \
  --weights data/imagenet_models/vgg16.npy \
  --imdb rgbd_scene_train \
  --cfg experiments/cfgs/rgbd_scene_single_color_fcn8.yml \
  --iters 40000

if [ -f $PWD/output/rgbd_scene/rgbd_scene_val/fcn8_color_single_frame_rgbd_scene_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/rgbd_scene/rgbd_scene_val/fcn8_color_single_frame_rgbd_scene_iter_40000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network fcn8_vgg \
  --weights data/imagenet_models/vgg16.npy \
  --model output/rgbd_scene/rgbd_scene_train/fcn8_color_single_frame_rgbd_scene_iter_40000.ckpt \
  --imdb rgbd_scene_val \
  --cfg experiments/cfgs/rgbd_scene_single_color_fcn8.yml
