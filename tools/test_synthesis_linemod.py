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
from synthesize import synthesizer
import scipy.io
import cv2
import numpy as np

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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    which_class = 7
    classes_all = ('ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher')

    num_images = 80000
    height = 480
    width = 640
    fx = 572.41140
    fy = 573.57043
    px = 325.26110
    py = 242.04899
    zfar = 6.0
    znear = 0.25;
    factor_depth = 1000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    root = '/home/yuxiang/Datasets/LINEMOD_Dataset/data_syn_lighting/' + classes_all[which_class] + '/'

    if not os.path.exists(root):
        os.makedirs(root)

    synthesizer_ = synthesizer.PySynthesizer(args.cad_name, args.pose_name)
    synthesizer_.setup(width, height)

    extent_file = '/home/yuxiang/Projects/Deep_Pose/data/LINEMOD/extents.txt'
    extents = np.zeros((9, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_file).astype(np.float32)
    print extents

    i = 0
    while i < num_images:
        
        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.uint8)
        depth_syn = np.zeros((height, width), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        poses = np.zeros((1, 7), dtype=np.float32)
        centers = np.zeros((1, 2), dtype=np.float32)
        synthesizer_.render_one(im_syn, depth_syn, vertmap_syn, poses, centers, extents, fx, fy, px, py, znear, zfar, int(which_class))
        im_syn = im_syn[::-1, :, :]
        depth_syn = depth_syn[::-1, :]

        # convert depth
        im_depth_raw = factor_depth * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * depth_syn - 1))
        I = np.where(depth_syn == 1)
        im_depth_raw[I[0], I[1]] = 0

        # compute labels from vertmap
        label = np.round(vertmap_syn[:, :, 0]) + 1
        label[np.isnan(label)] = 0

        I = np.where(label != which_class + 1)
        label[I[0], I[1]] = 0

        I = np.where(label == which_class + 1)
        if len(I[0]) < 800:
            continue

        # convert pose
        qt = np.zeros((3, 4, 1), dtype=np.float32)
        qt[:, :3, 0] = quat2mat(poses[0, :4])
        qt[:, 3, 0] = poses[0, 4:]

        # process the vertmap
        vertmap_syn[:, :, 0] = vertmap_syn[:, :, 0] - np.round(vertmap_syn[:, :, 0])
        vertmap_syn[np.isnan(vertmap_syn)] = 0

        # metadata
        metadata = {'poses': qt, 'center': centers, 'vertmap': vertmap_syn, \
                    'cls_indexes': which_class + 1, 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}

        # save image
        filename = root + '{:06d}-color.png'.format(i)
        cv2.imwrite(filename, im_syn)

        # save depth
        filename = root + '{:06d}-depth.png'.format(i)
        cv2.imwrite(filename, im_depth_raw.astype(np.uint16))

        # save label
        filename = root + '{:06d}-label.png'.format(i)
        cv2.imwrite(filename, label.astype(np.uint8))

        # save meta_data
        filename = root + '{:06d}-meta.mat'.format(i)
        scipy.io.savemat(filename, metadata, do_compression=True)
        print filename

        i += 1
