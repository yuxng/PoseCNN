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
import libsynthesizer
import scipy.io
import cv2
import numpy as np
import cPickle

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

    which_class = 15
    classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                   '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                   '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                   '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    num_classes = len(classes_all)

    num_images = 10000
    height = 480
    width = 640
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    zfar = 6.0
    znear = 0.25;
    factor_depth = 1000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    root = '/capri/YCB_Video_Dataset/data_syn_' + classes_all[which_class] + '/'

    if not os.path.exists(root):
        os.makedirs(root)

    synthesizer_ = libsynthesizer.Synthesizer(args.cad_name, args.pose_name)
    synthesizer_.setup(width, height)
    synthesizer_.init_rand(1200)

    extent_file = '/home/yuxiang/Projects/Deep_Pose/data/LOV/extents.txt'
    extents = np.zeros((num_classes + 1, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_file).astype(np.float32)
    print extents

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
        poses = np.zeros((1, 7), dtype=np.float32)
        centers = np.zeros((1, 2), dtype=np.float32)
        synthesizer_.render_one_python(int(which_class), int(width), int(height), fx, fy, px, py, znear, zfar, im_syn, depth_syn, vertmap_syn, poses, centers, extents)

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
        metadata = {'poses': qt, 'center': centers, \
                    'cls_indexes': which_class + 1, 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}

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
        scipy.io.savemat(filename, metadata, do_compression=True)
        print filename

        i += 1
