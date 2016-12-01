#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import scipy.io
from utils.voxelizer import Voxelizer, set_axes_equal
from kinect_fusion import kfusion

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
    voxelizer = Voxelizer(128, 7)

    # kinect fusion
    KF = kfusion.PyKinectFusion(args.rig_name)

    video_index = ''
    transformations = []
    cameras = []
    RTs = []
    have_prediction = False
    for i in xrange(num_images):
        print i
        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            voxelizer.reset()
            have_prediction = False
        else:
            if video_index != image_index[:pos]:
                # show the 3D points
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(temp_points[:,0], temp_points[:,1], temp_points[:,2], c='r', marker='o')
                print 'X:', voxelizer.min_x, voxelizer.max_x
                print 'Y:', voxelizer.min_y, voxelizer.max_y
                print 'Z:', voxelizer.min_z, voxelizer.max_z

                # show the camera positions
                ax.scatter(0, 0, 0, c='g', marker='o')
                for j in range(len(transformations)):
                    t = transformations[j]
                    ax.scatter(t[0, 3], t[1, 3], t[2, 3], c='g', marker='o')
                    RT = RTs[0]
                    C = cameras[j+1]
                    C1 = np.dot(RT[0:3, 0:3], np.transpose(C)) + RT[0:3, 3].reshape((3,1))
                    ax.scatter(C1[0, 0], C1[1, 0], C1[2, 0], c='b', marker='o')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                set_axes_equal(ax)
                plt.show()

                video_index = image_index[:pos]
                print 'start video {}'.format(video_index)
                voxelizer.reset()
                have_prediction = False
                KF.reset()
                transformations = []
                cameras = []
                RTs = []

        # RGB image
        im = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)

        # depth image
        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))
        RTs.append(meta_data['rotation_translation_matrix'])
        cameras.append(meta_data['camera_location'])

        # backprojection for the first frame
        if not have_prediction:
            points = voxelizer.backproject_camera(im_depth, meta_data)
            voxelizer.voxelize(points)
            KF.set_voxel_grid(voxelizer.min_x, voxelizer.min_y, voxelizer.min_z, voxelizer.max_x-voxelizer.min_x, voxelizer.max_y-voxelizer.min_y, voxelizer.max_y-voxelizer.min_y)

            # sample template points
            X = points[0,:]
            Y = points[1,:]
            Z = points[2,:]
            index = np.where(np.isfinite(X))[0]
            perm = np.random.permutation(np.arange(len(index)))
            num_temp = min(10000, len(index))
            index = index[perm[:num_temp]]
            temp_points = np.zeros((num_temp, 3), dtype=np.float64)
            temp_points[:,0] = X[index]
            temp_points[:,1] = Y[index]
            temp_points[:,2] = Z[index]

        # run kinect fusion
        KF.feed_data(im_depth, im, im.shape[1], im.shape[0])
        KF.back_project();
        if have_prediction:
          pose = KF.solve_pose()
          transformations.append(pose)
        KF.fuse_depth()
        KF.extract_surface()
        KF.render()
        have_prediction = True
        KF.draw()
