#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from synthesize import synthesizer
import argparse
import os, sys
from transforms3d.quaternions import quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np

if __name__ == '__main__':

    num_seqs = 15
    num_images = np.array([1236, 1214, 1233, 1201, 1196, \
    1179, 1240, 1188, 1254, 1253, \
    1220, 1237, 1152, 1227, 1243]);

    root = '/home/yuxiang/Projects/Deep_Pose/data/LINEMOD/data/'

    for k in xrange(num_seqs):
        for i in xrange(num_images[k]):
            '''
            # color
            filename = root + '{:04d}/{:06d}-color.png'.format(k, i+1)
            print filename
            rgba = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # depth
            filename = root + '{:04d}/{:06d}-depth.png'.format(k, i+1)
            print filename
            depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # label
            filename = root + '{:04d}/{:06d}-label.png'.format(k, i+1)
            print filename
            label = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            '''
            # save meta_data
            filename = root + '{:02d}/{:06d}-meta.mat'.format(k+1, i+1)
            print filename
            meta_data = scipy.io.loadmat(filename)
