#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/shapenet_scene_vgg16_depth.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#time ./tools/train_net.py --gpu $1 \
#  --network vgg16 \
#  --weights data/imagenet_models/vgg16_convs.npy \
#  --imdb shapenet_scene_train \
#  --cfg experiments/cfgs/shapenet_scene.yml \
#  --iters 40000

time ./tools/test_net.py --gpu $1 \
  --network vgg16 \
  --model output/shapenet_scene/shapenet_scene_train/vgg16_fcn_depth_shapenet_scene_iter_30000.ckpt \
  --imdb shapenet_scene_val \
  --cfg experiments/cfgs/shapenet_scene.yml

# create output video
#/var/Softwares/ffmpeg-3.1.3-64bit-static/ffmpeg -r 8 -start_number 0 \
#  -i /var/Projects/FCN/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_depth_shapenet_scene_iter_40000/images/%04d.png -vcodec mpeg4 -b 800K \
#  /var/Projects/FCN/output/shapenet_scene/shapenet_scene_val/vgg16_fcn_depth_shapenet_scene_iter_40000/results.avi
