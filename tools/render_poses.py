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
from transforms3d.quaternions import mat2quat, quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import libsynthesizer
import scipy.io
import cv2
import numpy as np
from utils.se3 import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--ckpt', dest='pretrained_ckpt',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    # load model transforms
    filename = '/capri/YCB/model_transforms.mat'
    mobject = scipy.io.loadmat(filename)
    model_transforms = mobject['model_transforms']

    height = 480
    width = 640
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    zfar = 6.0
    znear = 0.25
    num_classes = 22
    factor_depth = 10000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    root = '/capri/YCB/data_lov/010_potted_meat_can/'
    num_images = 50000
    #root = '/capri/YCB/data_fat/'
    #num_images = 61500
    is_show = 0

    if not os.path.exists(root):
        os.makedirs(root)

    synthesizer_ = libsynthesizer.Synthesizer(args.cad_name, args.pose_name)
    synthesizer_.setup(width, height)

    parameters = np.zeros((6, ), dtype=np.float32)
    parameters[0] = fx
    parameters[1] = fy
    parameters[2] = px
    parameters[3] = py
    parameters[4] = znear
    parameters[5] = zfar

    if is_show:
        perm = np.random.permutation(np.arange(num_images))
    else:
        perm = xrange(num_images)

    for i in perm:

        # load meta data
        filename = root + '{:06d}-meta.mat'.format(i)
        meta_data = scipy.io.loadmat(filename)

        # prepare data
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        num = poses.shape[2]
        channel = 8
        qt = np.zeros((num, channel), dtype=np.float32)
        for j in xrange(num):
            class_id = int(meta_data['cls_indexes'][j]) - 1
            RT = se3_mul(poses[:,:,j], model_transforms[:,:,class_id])

            R = RT[:, :3]
            T = RT[:, 3]
            qt[j, 0] = meta_data['cls_indexes'][j]
            qt[j, 1:5] = mat2quat(R)
            qt[j, 5:] = T
        
        # render a synthetic image
        im_syn = np.zeros((height, width, 3), dtype=np.uint8)
        synthesizer_.render_poses_python(int(num), int(channel), int(width), int(height), parameters, im_syn, qt)

        # convert images
        im_syn = im_syn[::-1, :, 0]

        # show images
        if is_show:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            filename = root + '{:06d}.png'.format(i)
            im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            im = im[:, :, (2, 1, 0)]
            plt.imshow(im)
            ax.set_title('color') 

            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(im_syn)
            ax.set_title('render') 
            plt.show()
        else:
            # save image
            filename = root + '{:06d}-object.png'.format(i)
            cv2.imwrite(filename, im_syn)
            print filename
