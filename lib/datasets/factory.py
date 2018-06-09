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
import datasets.lov_single
import datasets.ycb
import datasets.ycb_single
import datasets.yumi
import datasets.linemod
import datasets.sym
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
for split in ['train', 'val', 'keyframe', 'trainval', 'debug', 'train_few', 'val_few']:
    name = 'lov_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.lov(split))

for cls in ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']:
    for split in ['train', 'val', 'keyframe']:
        name = 'lov_single_{}_{}'.format(cls, split)
        print name
        __sets[name] = (lambda cls=cls, split=split:
                datasets.lov_single(cls, split))

# ycb dataset
for split in ['trainval']:
    name = 'ycb_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.ycb(split))

for cls in ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']:
    for split in ['train']:
        name = 'ycb_single_{}_{}'.format(cls, split)
        print name
        __sets[name] = (lambda cls=cls, split=split:
                datasets.ycb_single(cls, split))

# yumi dataset
for split in ['train']:
    name = 'yumi_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.yumi(split))

# linemod dataset
for cls in ['ape', 'benchvise', 'bowl', 'camera', 'can', \
    'cat', 'cup', 'driller', 'duck', 'eggbox', \
    'glue', 'holepuncher', 'iron', 'lamp', 'phone']:
    for split in ['train', 'test', 'train_few', 'test_few']:
        name = 'linemod_{}_{}'.format(cls, split)
        print name
        __sets[name] = (lambda cls=cls, split=split:
                datasets.linemod(cls, split))


# sym dataset
for split in ['train']:
    name = 'sym_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.sym(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
