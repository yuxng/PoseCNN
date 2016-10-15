# --------------------------------------------------------
# FCN
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.vgg16
import tensorflow as tf

__sets['vgg16'] = networks.vgg16()

def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
