#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_synthesis_linemod.py --gpu 0 \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt
