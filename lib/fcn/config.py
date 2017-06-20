# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""FCN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
import math
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fcn.config import cfg
cfg = __C

__C.FLIP_X = False
__C.INPUT = 'RGBD'
__C.NETWORK = 'VGG16'
__C.LOSS_FUNC = 'not_specified'

#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.SINGLE_FRAME = False
__C.TRAIN.TRAINABLE = True
__C.TRAIN.VERTEX_REG = False
__C.TRAIN.VERTEX_W = 10.0
__C.TRAIN.VISUALIZE = False
__C.TRAIN.GAN = False

# learning rate
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.LEARNING_RATE_ADAM = 0.1
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 30000

# voxel grid size
__C.TRAIN.GRID_SIZE = 256

# Scales to compute real features
__C.TRAIN.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)

# parameters for data augmentation
__C.TRAIN.CHROMATIC = True

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2
__C.TRAIN.NUM_STEPS = 5
__C.TRAIN.NUM_UNITS = 64
__C.TRAIN.NUM_CLASSES = 10

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'caffenet_fast_rcnn'
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TRAIN.DISPLAY = 20
__C.TRAIN.OPTICAL_FLOW = False
__C.TRAIN.DELETE_OLD_CHECKPOINTS = False
__C.TRAIN.VISUALIZE_DURING_TRAIN = False
__C.TRAIN.OPTIMIZER = 'MomentumOptimizer'


__C.NET_CONF = edict()
__C.NET_CONF.COMBINE_CONVOLUTION_SIZE = 1
__C.NET_CONF.CONCAT_OR_SUBTRACT = "concat"
__C.NET_CONF.N_CONVOLUTIONS = 1


#
# Testing options
#

__C.TEST = edict()

__C.TEST.SINGLE_FRAME = False
__C.TEST.VERTEX_REG = False
__C.TEST.VISUALIZE = False
__C.TEST.RANSAC = False
__C.TEST.GAN = False
__C.TEST.OPTICAL_FLOW = False

# Scales to compute real features
__C.TEST.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)

# voxel grid size
__C.TEST.GRID_SIZE = 256

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Default GPU device id
__C.GPU_ID = 0

def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
