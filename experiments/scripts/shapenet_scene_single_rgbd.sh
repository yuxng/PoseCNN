#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/shapenet_scene_single_rgbd.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frame
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb shapenet_scene_train \
  --cfg experiments/cfgs/shapenet_scene_single_rgbd.yml \
  --iters 40000

if [ -f $PWD/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_rgbd_single_frame_shapenet_scene_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_rgbd_single_frame_shapenet_scene_iter_40000/segmentations.pkl
fi

# test the single frame network
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/shapenet_scene/shapenet_scene_train/vgg16_fcn_rgbd_single_frame_shapenet_scene_iter_40000.ckpt \
  --imdb shapenet_scene_val \
  --cfg experiments/cfgs/shapenet_scene_single_rgbd.yml
