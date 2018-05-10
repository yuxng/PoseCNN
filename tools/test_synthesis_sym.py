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
import cPickle
import libsynthesizer

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
    cfg.BACKGROUND = args.background_name

    num_images = 100
    height = 480
    width = 640
    fx = 572.41140
    fy = 573.57043
    px = 325.26110
    py = 242.04899
    zfar = 6.0
    znear = 0.25;
    tnear = 2.0
    tfar = 5.0
    num_classes = 1 + 1
    factor_depth = 1000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    root = '/home/yuxiang/Projects/Deep_Pose/data/SYM/data_syn/'

    if not os.path.exists(root):
        os.makedirs(root)

    synthesizer_ = libsynthesizer.Synthesizer(args.cad_name, args.pose_name)
    synthesizer_.setup(width, height)
    synthesizer_.init_rand(1200)

    parameters = np.zeros((8, ), dtype=np.float32)
    parameters[0] = fx
    parameters[1] = fy
    parameters[2] = px
    parameters[3] = py
    parameters[4] = znear
    parameters[5] = zfar
    parameters[6] = tnear
    parameters[7] = tfar

    # load background
    cache_file = cfg.BACKGROUND
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            backgrounds = cPickle.load(fid)
        print 'backgrounds loaded from {}'.format(cache_file)

    i = 0
    while i < num_images:

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        class_indexes = -1 * np.ones((num_classes, ), dtype=np.float32)
        poses = np.zeros((num_classes, 7), dtype=np.float32)
        centers = np.zeros((num_classes, 2), dtype=np.float32)
        is_sampling = False
        is_sampling_pose = False
        synthesizer_.render_python(int(width), int(height), parameters, \
                                   im_syn, depth_syn, vertmap_syn, class_indexes, poses, centers, is_sampling, is_sampling_pose)

        # convert images
        im_syn = np.clip(255 * im_syn, 0, 255)
        im_syn = im_syn.astype(np.uint8)
        depth_syn = depth_syn[:, :, 0]

        # convert depth
        im_depth_raw = factor_depth * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * depth_syn - 1))
        I = np.where(depth_syn == 1)
        im_depth_raw[I[0], I[1]] = 0

        # compute labels from vertmap
        label = np.round(vertmap_syn[:, :, 0]) + 1
        label[np.isnan(label)] = 0

        flag = 1
        for j in xrange(1, num_classes):
            I = np.where(label == j)
            if len(I[0]) < 800:
                flag = 0
                break
        if flag == 0:
            continue

        # convert pose
        index = np.where(class_indexes >= 0)[0]
        num = len(index)
        qt = np.zeros((3, 4, num), dtype=np.float32)
        for j in xrange(num):
            ind = index[j]
            qt[:, :3, j] = quat2mat(poses[ind, :4])
            qt[:, 3, j] = poses[ind, 4:]

        # process the vertmap
        vertmap_syn[:, :, 0] = vertmap_syn[:, :, 0] - np.round(vertmap_syn[:, :, 0])
        vertmap_syn[np.isnan(vertmap_syn)] = 0

        # metadata
        metadata = {'poses': qt, 'center': centers[class_indexes[index].astype(int), :], 'vertmap': vertmap_syn, \
                    'cls_indexes': class_indexes[index] + 1, 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}


        # sample a background image
        rgba = im_syn
        ind = np.random.randint(len(backgrounds), size=1)[0]
        filename = backgrounds[ind]
        background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        try:
            background = cv2.resize(background, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)
        except:
            if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
                background = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint16)
            else:
                background = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
            print 'bad background image'

        if cfg.INPUT != 'DEPTH' and cfg.INPUT != 'NORMAL' and len(background.shape) != 3:
            background = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
            print 'bad background image'

        # add background
        im = np.copy(rgba[:,:,:3])
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
            im_depth[I[0], I[1]] = background[I[0], I[1]] / 10
        else:
            im[I[0], I[1], :] = background[I[0], I[1], :3]

        # save image
        filename = root + '{:06d}-color.png'.format(i)
        cv2.imwrite(filename, im)

        # save depth
        filename = root + '{:06d}-depth.png'.format(i)
        cv2.imwrite(filename, im_depth_raw.astype(np.uint16))

        # save label
        filename = root + '{:06d}-label.png'.format(i)
        cv2.imwrite(filename, label.astype(np.uint8))

        # save meta_data
        filename = root + '{:06d}-meta.mat'.format(i)
        print filename
        scipy.io.savemat(filename, metadata, do_compression=True)

        i += 1
