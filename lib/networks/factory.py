# --------------------------------------------------------
# FCN
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.vgg16
import networks.vgg16_convs
import networks.vgg16_full
import networks.vgg16_det
import networks.vgg16_gan
import networks.dcgan
import networks.resnet50
import tensorflow as tf
from fcn.config import cfg

if cfg.TRAIN.SINGLE_FRAME:
    if cfg.NETWORK == 'VGG16':
        __sets['vgg16_convs'] = networks.vgg16_convs(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, \
                                                     cfg.TRAIN.THRESHOLD_LABEL, cfg.TRAIN.VOTING_THRESHOLD, \
                                                     cfg.TRAIN.VERTEX_REG_2D, cfg.TRAIN.VERTEX_REG_3D, \
                                                     cfg.TRAIN.POSE_REG, cfg.TRAIN.ADAPT, cfg.TRAIN.TRAINABLE, cfg.IS_TRAIN)
    if cfg.NETWORK == 'VGG16FULL':
        __sets['vgg16_full'] = networks.vgg16_full(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, \
                                                     cfg.TRAIN.VERTEX_REG_2D, cfg.TRAIN.VERTEX_REG_3D, cfg.TRAIN.POSE_REG, \
                                                     cfg.TRAIN.MATCHING, cfg.TRAIN.TRAINABLE, cfg.IS_TRAIN)
    if cfg.NETWORK == 'VGG16DET':
        __sets['vgg16_det'] = networks.vgg16_det(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.FEATURE_STRIDE, cfg.ANCHOR_SCALES, \
                                                 cfg.ANCHOR_RATIOS, cfg.TRAIN.TRAINABLE, cfg.IS_TRAIN)
    if cfg.NETWORK == 'VGG16GAN':
        __sets['vgg16_gan'] = networks.vgg16_gan(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, \
                                                 cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    if cfg.NETWORK == 'DCGAN':
        __sets['dcgan'] = networks.dcgan()
    if cfg.NETWORK == 'RESNET50':
        __sets['resnet50'] = networks.resnet50(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.SCALES_BASE)
    if cfg.NETWORK == 'FCN8VGG':
        __sets['fcn8_vgg'] = networks.fcn8_vgg(cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.MODEL_PATH)
else:
    __sets['vgg16'] = networks.vgg16(cfg.INPUT, cfg.TRAIN.NUM_STEPS, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE)

def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown network: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
