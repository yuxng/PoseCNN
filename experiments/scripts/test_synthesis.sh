#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_synthesis.py --gpu 0 \
  --cad data/LOV/models_full.txt \
  --pose data/LOV/poses.txt \
  --background data/cache/backgrounds.pkl
