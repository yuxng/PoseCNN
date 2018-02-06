#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_benchvise_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export LD_PRELOAD=/usr/lib/libtcmalloc.so.4

# train for labeling
#time ./tools/train_net.py --gpu 0 \
#  --network vgg16_convs \
#  --weights data/imagenet_models/vgg16_convs.npy \
#  --imdb linemod_benchvise_train_few \
#  --cfg experiments/cfgs/linemod_benchvise.yml \
#  --iters 40000

# train for pose
time ./tools/train_net.py --gpu 0 \
  --network vgg16_convs \
  --weights data/imagenet_models/vgg16.npy \
  --ckpt output/linemod/linemod_benchvise_train_few/vgg16_fcn_color_single_frame_linemod_benchvise_iter_40000.ckpt \
  --imdb linemod_benchvise_train_few \
  --cfg experiments/cfgs/linemod_benchvise_pose.yml \
  --iters 80000
