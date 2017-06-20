#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a Fully Convolutional Network (FCN) on image segmentation database."""

import _init_paths
import cv2
from fcn.train import get_training_roidb, train_net, train_gan, train_flow
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
import datetime
import os
import sys

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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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
    roidb = get_training_roidb(imdb)

    if not cfg.TRAIN.OPTICAL_FLOW:
        output_dir = get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)
    else:
        output_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR,
                                          "batch_size_" + str(cfg.TRAIN.IMS_PER_BATCH) + "_loss_" + str(cfg.LOSS_FUNC) +
                                          "_optimizer_" + cfg.TRAIN.OPTIMIZER + "_conv_size_" + str(cfg.NET_CONF.COMBINE_CONVOLUTION_SIZE) +
                                          "_ConcatSub_" + cfg.NET_CONF.CONCAT_OR_SUBTRACT + "_" + str(datetime.date.today())))
        print 'Output will be saved to `{:s}`'.format(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # sys.stderr.write("output directory already exists, please move it or choose another output dir")
            pass
        pprint.pprint(cfg, stream=open(output_dir + "/config.txt", 'w'))

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    print device_name

    if cfg.NETWORK == 'FCN8VGG':
        path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
        cfg.TRAIN.MODEL_PATH = path
        pretrained_model = None
    else:
        pretrained_model = args.pretrained_model

    from networks.factory import get_network
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    if cfg.TRAIN.GAN:
        train_gan(network, imdb, roidb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=args.max_iters)
    elif cfg.TRAIN.OPTICAL_FLOW:
        train_flow(network, imdb, roidb, output_dir,
                  pretrained_model=pretrained_model,
                  max_iters=args.max_iters)
    else:
        train_net(network, imdb, roidb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=args.max_iters)
