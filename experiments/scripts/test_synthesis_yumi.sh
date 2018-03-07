#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_synthesis_yumi.py --gpu 0 \
  --cad data/Yumi/models.txt \
  --pose data/Yumi/poses.txt
