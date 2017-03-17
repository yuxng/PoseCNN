#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/shapenet_single_single_color.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb shapenet_single_train \
  --cfg experiments/cfgs/shapenet_single_single_color.yml \
  --iters 20000

if [ -f $PWD/output/shapenet_single/shapenet_single_val/vgg16_fcn_color_single_frame_shapenet_single_iter_20000/segmentations.pkl ]
then
  rm $PWD/output/shapenet_single/shapenet_single_val/vgg16_fcn_color_single_frame_shapenet_single_iter_20000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/shapenet_single/shapenet_single_train/vgg16_fcn_color_single_frame_shapenet_single_iter_20000.ckpt \
  --imdb shapenet_single_val \
  --cfg experiments/cfgs/shapenet_single_single_color.yml \
  --rig data/LOV/camera.json
