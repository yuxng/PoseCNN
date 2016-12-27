#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from kinect_fusion import kfusion
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
#from utils.voxelizer import Voxelizer, set_axes_equal
import time
from utils.se3 import *
import cv2
import numpy as np
# import scipy.io

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='shapenet_scene_val', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
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

    # imdb
    imdb = get_imdb(args.imdb_name)

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)

    # voxelizer
    #voxelizer = Voxelizer(128, imdb.num_classes)
    labels_voxel = np.zeros((128, 128, 128), dtype=np.int32) 

    # kinect fusion
    KF = kfusion.PyKinectFusion(args.rig_name)

    # construct colors
    colors = np.zeros((3, imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[0, i] = 255 * imdb._class_colors[i][0]
        colors[1, i] = 255 * imdb._class_colors[i][1]
        colors[2, i] = 255 * imdb._class_colors[i][2]
    colors[:,0] = 255

    video_index = ''
    have_prediction = False
    restart = False
    RTs = []
    for i in xrange(num_images):
        print i
        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            #voxelizer.reset()
            have_prediction = False
            restart = False
        else:
            if video_index != image_index[:pos]:
                video_index = image_index[:pos]
                print 'start video {}'.format(video_index)
                #voxelizer.reset()
                have_prediction = False
                restart = False
                KF.reset()
                RTs = []

        # RGB image
        im = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
        tmp = im.copy()
        tmp[:,:,0] = im[:,:,2]
        tmp[:,:,2] = im[:,:,0]
        im = tmp

        # depth image
        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        # load meta data
        # meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))
        # RTs.append(meta_data['rotation_translation_matrix'])

        # backprojection for the first frame
        # points = voxelizer.backproject_camera(im_depth, meta_data)
        if not have_prediction:    
            # voxelizer.voxelize(points)
            # KF.set_voxel_grid(voxelizer.min_x, voxelizer.min_y, voxelizer.min_z, voxelizer.max_x-voxelizer.min_x, voxelizer.max_y-voxelizer.min_y, voxelizer.max_z-voxelizer.min_z)
            KF.set_voxel_grid(-1.032, -1.713, -0.2234, 2.5, 2.5, 2.5)

        # run kinect fusion
        KF.feed_data(im_depth, im, im.shape[1], im.shape[0], 10000.0) #float(meta_data['factor_depth']))
        KF.back_project();
        if have_prediction:
            pose_world2live, pose_live2world = KF.solve_pose()
        else:
            pose_world2live = np.zeros((3,4), dtype=np.float32)
            pose_world2live[0, 0] = 1
            pose_world2live[1, 1] = 1
            pose_world2live[2, 2] = 1
            pose_live2world = pose_world2live

        # check if points outside voxel space
        #flag = voxelizer.check_points(points, pose_live2world)
        #if not flag:
        #    print 'points outside voxel space, restart from next frame'
        #    restart = True

        '''
        # compute pose from GT for comparison
        RT_world = RTs[0]
        RT_live = RTs[-1]
        RT_world2live = se3_mul(RT_live, se3_inverse(RT_world))
        RT_live2world = se3_inverse(RT_world2live)
        print "pose_world2live:"
        print pose_world2live
        print RT_world2live
        print "pose_live2world:"
        print pose_live2world
        print RT_live2world
        # time.sleep(3)
        '''

        KF.fuse_depth()
        print 'finish fuse depth'
        KF.extract_surface()
        print 'finish extract surface'
        KF.render()
        print 'finish render'
        KF.feed_label(im, labels_voxel, colors)
        print 'finish feed label'
        KF.draw()
        print 'finish draw'

        have_prediction = True
