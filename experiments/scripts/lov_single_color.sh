#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_single_color.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
 export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb lov_train \
  --cfg experiments/cfgs/lov_single_color.yml \
  --cad data/LOV/models.txt \
  --iters 40000

#if [ -f $PWD/output/lov/lov_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl ]
#then
#  rm $PWD/output/lov/lov_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl
#fi

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output/lov/lov_train/vgg16_fcn_color_single_frame_synthesize_lov.ckpt \
  --imdb lov_val \
  --cfg experiments/cfgs/lov_single_color.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/lov_train_backgrounds.pkl
