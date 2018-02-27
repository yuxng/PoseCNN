#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_duck_det_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
# test for semantic labeling
time ./tools/test_net.py --gpu 0 \
  --network vgg16_det \
  --model output/linemod/linemod_duck_train_few/vgg16_fcn_detection_linemod_duck_iter_160000.ckpt \
  --imdb linemod_duck_test_few \
  --cfg experiments/cfgs/linemod_duck_det.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --background data/cache/backgrounds.pkl
