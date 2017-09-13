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

    num_seqs = 92
    num_images = np.array([762, 1112, 1719, 2299, 2172, 1506, 1626, 2018, 2991, 1591, 1898, \
    1107, 1104, 1800, 1619, 2305, 1335, 1716, 1424, 2218, 1873, 731, 1153, 1618, \
    1401, 1444, 1607, 1448, 1558, 1164, 1124, 1952, 1254, 1567, 1554, 1668, \
    2512, 2157, 3467, 3160, 2393, 2450, 2122, 2591, 2542, 2509, 2033, 2089, \
    2244, 2402, 1917, 2009, 900, 837, 1929, 1830, 1226, 1872, 1720, 1864, \
    754, 533, 680, 667, 668, 653, 801, 849, 884, 784, 1016, 951, 890, 719, 908, \
    694, 864, 779, 689, 789, 788, 985, 743, 953, 986, 890, 897, 948, 453, 868, 842, 890]) - 1;

    root = '/home/yuxiang/mnt1/yuxiang/LOV_Dataset/data/'

    for k in xrange(68, num_seqs):
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
            filename = root + '{:04d}/{:06d}-meta.mat'.format(k, i+1)
            print filename
            meta_data = scipy.io.loadmat(filename)
