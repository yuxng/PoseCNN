#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
import argparse
import os, sys
from transforms3d.quaternions import quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np

if __name__ == '__main__':

    num_images = 80000

    # which_class = 1
    # classes_all = ('ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher')
    # root = '/home/yuxiang/mnt1/yuxiang/LINEMOD_Dataset/data_syn/' + classes_all[which_class] + '/'
    root = '/capri/YCB_Video_Dataset/data_syn/'
    # root = '/nas-homes/yuxiang/LINEMOD_SIXD/data_syn/'

    for i in xrange(num_images):
        '''
        # color
        filename = root + '{:06d}-color.png'.format(i)
        print filename
        rgba = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # depth
        filename = root + '{:06d}-depth.png'.format(i)
        print filename
        depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # label
        filename = root + '{:06d}-label.png'.format(i)
        print filename
        label = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        '''
        # save meta_data
        filename = root + '{:06d}-meta.mat'.format(i)
        print filename
        meta_data = scipy.io.loadmat(filename)
