#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

LOG="experiments/logs/rgbd_scene_resnet50_single_frame.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
time ./tools/train_net.py --gpu 0 \
  --network resnet50 \
  --weights data/imagenet_models/resnet50.npy \
  --imdb rgbd_scene_train \
  --cfg experiments/cfgs/rgbd_scene_single_frame_resnet50.yml \
  --iters 80000


if [ -f $PWD/output/rgbd_scene/rgbd_scene_val/resnet50_fcn_rgbd_single_frame_rgbd_scene_iter_80000/segmentations.pkl ]
then
  rm $PWD/output/rgbd_scene/rgbd_scene_val/resnet50_fcn_rgbd_single_frame_rgbd_scene_iter_80000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network resnet50 \
  --model output/rgbd_scene/rgbd_scene_train/resnet50_fcn_rgbd_single_frame_rgbd_scene_iter_80000.ckpt \
  --imdb rgbd_scene_val \
  --cfg experiments/cfgs/rgbd_scene_single_frame_resnet50.yml \
  --rig data/RGBDScene/camera.json
