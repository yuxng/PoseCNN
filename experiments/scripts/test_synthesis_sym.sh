#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_synthesis_sym.py --gpu 0 \
  --cad data/SYM/models.txt \
  --pose data/SYM/poses.txt \
  --background data/cache/backgrounds.pkl
