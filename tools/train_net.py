#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a Fully Convolutional Network (FCN) on image segmentation database."""

import _init_paths
from fcn.train import get_training_roidb, train_net, train_net_det
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
import tensorflow as tf
import threading
from Queue import Queue
import cv2

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


def render_one(data_queue, intrinsic_matrix, extents, points):

    synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
    synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    which_class = cfg.TRAIN.SYN_CLASS_INDEX
    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.25;
    factor_depth = 1000.0

    while True:

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        poses = np.zeros((1, 7), dtype=np.float32)
        centers = np.zeros((1, 2), dtype=np.float32)
        synthesizer.render_one_python(int(which_class), int(width), int(height), fx, fy, px, py, znear, zfar, \
            im_syn, depth_syn, vertmap_syn, poses, centers, extents)

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

        # compute box
        x3d = np.ones((4, points.shape[1]), dtype=np.float32)
        cls = 1
        x3d[0, :] = points[cls,:,0]
        x3d[1, :] = points[cls,:,1]
        x3d[2, :] = points[cls,:,2]
        RT = qt[:, :, 0]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        box = np.zeros((1, 4), dtype=np.float32)
        box[0, 0] = np.min(x2d[0, :])
        box[0, 1] = np.min(x2d[1, :])
        box[0, 2] = np.max(x2d[0, :])
        box[0, 3] = np.max(x2d[1, :])

        # metadata
        metadata = {'poses': qt, 'center': centers, 'box': box, \
                    'cls_indexes': np.array([which_class + 1]), 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}

        # construct data
        data = {'image': im_syn, 'depth': im_depth_raw.astype(np.uint16), 'label': label.astype(np.uint8), 'meta_data': metadata}
        data_queue.put(data)


def render(data_queue, intrinsic_matrix, points):

    synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
    synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    height = cfg.TRAIN.SYN_HEIGHT
    width = cfg.TRAIN.SYN_WIDTH
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.25;
    tnear = cfg.TRAIN.SYN_TNEAR
    tfar = cfg.TRAIN.SYN_TFAR
    factor_depth = 1000.0
    num_classes = points.shape[0]

    parameters = np.zeros((8, ), dtype=np.float32)
    parameters[0] = fx
    parameters[1] = fy
    parameters[2] = px
    parameters[3] = py
    parameters[4] = znear
    parameters[5] = zfar
    parameters[6] = tnear
    parameters[7] = tfar

    while True:

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        class_indexes = -1 * np.ones((num_classes, ), dtype=np.float32)
        poses = np.zeros((num_classes, 7), dtype=np.float32)
        centers = np.zeros((num_classes, 2), dtype=np.float32)
        is_sampling = cfg.TRAIN.SYN_SAMPLE_OBJECT
        is_sampling_pose = cfg.TRAIN.SYN_SAMPLE_POSE
        synthesizer.render_python(int(width), int(height), parameters, \
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

        # convert pose
        index = np.where(class_indexes >= 0)[0]
        num = len(index)
        qt = np.zeros((3, 4, num), dtype=np.float32)
        for j in xrange(num):
            ind = index[j]
            qt[:, :3, j] = quat2mat(poses[ind, :4])
            qt[:, 3, j] = poses[ind, 4:]

        flag = 1
        for j in xrange(num):
            cls = class_indexes[index[j]] + 1
            I = np.where(label == cls)
            if len(I[0]) < 800:
                flag = 0
                break
        if flag == 0:
            continue

        # process the vertmap
        vertmap_syn[:, :, 0] = vertmap_syn[:, :, 0] - np.round(vertmap_syn[:, :, 0])
        vertmap_syn[np.isnan(vertmap_syn)] = 0

        # compute box
        box = np.zeros((num, 4), dtype=np.float32)
        for j in xrange(num):
            cls = int(class_indexes[index[j]]) + 1
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]
            RT = qt[:, :, j]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
            box[j, 0] = np.min(x2d[0, :])
            box[j, 1] = np.min(x2d[1, :])
            box[j, 2] = np.max(x2d[0, :])
            box[j, 3] = np.max(x2d[1, :])

        # metadata
        metadata = {'poses': qt, 'center': centers[class_indexes[index].astype(int), :], 'box': box, \
                    'cls_indexes': class_indexes[index] + 1, 'intrinsic_matrix': intrinsic_matrix, 'factor_depth': factor_depth}

        # construct data
        data = {'image': im_syn, 'depth': im_depth_raw.astype(np.uint16), 'label': label.astype(np.uint8), 'meta_data': metadata}
        data_queue.put(data)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    print 'symmetry'
    print imdb._symmetry
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    print device_name

    if cfg.NETWORK == 'FCN8VGG':
        path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
        cfg.TRAIN.MODEL_PATH = path
        pretrained_model = None
    else:
        pretrained_model = args.pretrained_model

    cfg.RIG = args.rig_name
    cfg.CAD = args.cad_name
    cfg.POSE = args.pose_name
    cfg.IS_TRAIN = True

    if cfg.TRAIN.SYNTHESIZE and cfg.TRAIN.SYN_ONLINE:
        import libsynthesizer
        import scipy.io
        from transforms3d.quaternions import quat2mat

        # start rendering
        imdb.data_queue = Queue(maxsize=100)
        meta_data = scipy.io.loadmat(roidb[0]['meta_data'])
        intrinsic_matrix = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
        if cfg.TRAIN.SYN_CLASS_INDEX >= 0:
            t = threading.Thread(target=render_one, args=(imdb.data_queue, intrinsic_matrix, imdb._extents_all, imdb._points_all))
        else:
            t = threading.Thread(target=render, args=(imdb.data_queue, intrinsic_matrix, imdb._points_all))
        t.start()
    else:
        imdb.data_queue = []

    from networks.factory import get_network
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    if cfg.TRAIN.SEGMENTATION:
        train_net(network, imdb, roidb, output_dir,
                  pretrained_model=pretrained_model,
                  pretrained_ckpt=args.pretrained_ckpt,
                  max_iters=args.max_iters)
    else:
        train_net_det(network, imdb, roidb, output_dir,
                  pretrained_model=pretrained_model,
                  pretrained_ckpt=args.pretrained_ckpt,
                  max_iters=args.max_iters)
