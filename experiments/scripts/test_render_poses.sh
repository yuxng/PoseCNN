#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/render_poses.py --gpu 0 \
  --cad data/YCB/models.txt \
  --pose data/YCB/poses.txt
