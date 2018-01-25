#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_icp.py --gpu 0 \
  --cad /home/yuxiang/Datasets/LINEMOD_SIXD/models.txt \
  --pose /home/yuxiang/Datasets/LINEMOD_SIXD/poses.txt
