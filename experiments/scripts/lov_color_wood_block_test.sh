#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_color_wood_block_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# test FCN for single frames
time ./tools/test_net.py --gpu 0 \
  --network vgg16_convs \
  --model output_napoli/lov/lov_036_wood_block_train/vgg16_fcn_color_single_frame_2d_pose_add_nonsym_lov_wood_block_iter_160000.ckpt \
  --imdb lov_single_036_wood_block_train \
  --cfg experiments/cfgs/lov_color_wood_block.yml \
  --rig data/LOV/camera.json \
  --cad data/LOV/models.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
