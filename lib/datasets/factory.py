# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.shapenet_scene
import datasets.shapenet_single
import datasets.gmu_scene
import datasets.rgbd_scene
import datasets.lov
import datasets.linemod
import numpy as np

# shapenet dataset
for split in ['train', 'val']:
    name = 'shapenet_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.shapenet_scene(split))

for split in ['train', 'val']:
    name = 'shapenet_single_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.shapenet_single(split))

# gmu scene dataset
for split in ['train', 'val']:
    name = 'gmu_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.gmu_scene(split))

# rgbd scene dataset
for split in ['train', 'val', 'trainval']:
    name = 'rgbd_scene_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.rgbd_scene(split))

# lov dataset
for split in ['train', 'val', 'debug']:
    name = 'lov_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.lov(split))

# linemod dataset
for cls in ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']:
    for split in ['train', 'test']:
        name = 'linemod_{}_{}'.format(cls, split)
        print name
        __sets[name] = (lambda cls=cls, split=split:
                datasets.linemod(cls, split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
