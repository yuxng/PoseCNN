#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/rgbd_scene_multi_rgbd_reverse.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for multiple frames
time ./tools/train_net.py --gpu 0 \
  --network vgg16 \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb rgbd_scene_val \
  --cfg experiments/cfgs/rgbd_scene_multi_rgbd.yml \
  --iters 40000

if [ -f $PWD/output/rgbd_scene/rgbd_scene_train/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/rgbd_scene/rgbd_scene_train/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000/segmentations.pkl
fi

# test FCN for multiple frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16 \
  --model output/rgbd_scene/rgbd_scene_val/vgg16_fcn_rgbd_multi_frame_rgbd_scene_iter_40000.ckpt \
  --imdb rgbd_scene_train \
  --cfg experiments/cfgs/rgbd_scene_multi_rgbd.yml \
  --rig data/RGBDScene/camera.json
