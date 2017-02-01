#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/shapenet_scene_multi_depth.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for multiple frames
time ./tools/train_net.py --gpu 0 \
  --network vgg16 \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb shapenet_scene_train \
  --cfg experiments/cfgs/shapenet_scene_multi_depth.yml \
  --iters 40000

if [ -f $PWD/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_depth_multi_frame_shapenet_scene_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_depth_multi_frame_shapenet_scene_iter_40000/segmentations.pkl
fi

# test FCN for multiple frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16 \
  --model output/shapenet_scene/shapenet_scene_train/vgg16_fcn_depth_multi_frame_shapenet_scene_iter_40000.ckpt \
  --imdb shapenet_scene_val \
  --cfg experiments/cfgs/shapenet_scene_multi_depth.yml \
  --rig data/ShapeNetScene/camera.json
