#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/shapenet_single_single_color_gan.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
time ./tools/train_net.py --gpu 0 \
  --network dcgan \
  --imdb shapenet_single_train \
  --cfg experiments/cfgs/shapenet_single_single_color_gan.yml \
  --iters 20000

if [ -f $PWD/output/shapenet_single/shapenet_single_val/dcgan_color_single_frame_shapenet_single_iter_20000/segmentations.pkl ]
then
  rm $PWD/output/shapenet_single/shapenet_single_val/dcgan_color_single_frame_shapenet_single_iter_20000/segmentations.pkl
fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network dcgan \
  --model output/shapenet_single/shapenet_single_train/dcgan_color_single_frame_shapenet_single_iter_20000.ckpt \
  --imdb shapenet_single_val \
  --cfg experiments/cfgs/shapenet_single_single_color_gan.yml \
  --rig data/LOV/camera.json
