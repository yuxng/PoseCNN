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
from utils.voxelizer import set_axes_equal
from icp import icp_kernel

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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# backproject pixels into 3D points
def backproject_camera(im_depth, meta_data):

    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # get intrinsic matrix
    K = meta_data['intrinsic_matrix']
    K = np.matrix(K)
    Kinv = np.linalg.inv(K)

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the norm
    N = np.linalg.norm(R, axis=0)
        
    # normalization
    R = np.divide(R, np.tile(N, (3,1)))

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    # mask
    index = np.where(im_depth.flatten() == 0)
    X[:,index] = np.nan

    return np.array(X)

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # imdb
    imdb = get_imdb(args.imdb_name)

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)

    video_index = ''
    points_prev = np.zeros((3, 0), dtype=np.float32)
    transformations = []
    for i in xrange(num_images):
        print i
        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
        else:
            if video_index != image_index[:pos]:
                # show the camera positions
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(temp_points[:,0], temp_points[:,1], temp_points[:,2], c='r', marker='o')
                ax.scatter(0, 0, 0, c='g', marker='o')
                for j in range(len(transformations)):
                    t = transformations[j]
                    ax.scatter(t[0, 3], t[1, 3], t[2, 3], c='g', marker='o')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                set_axes_equal(ax)
                plt.show()

                video_index = image_index[:pos]
                print 'start video {}'.format(video_index)
                points_prev = np.zeros((3, 0), dtype=np.float32)
                transformations = []

        # RGB image
        rgba = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

        # depth image
        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        # backprojection
        points = backproject_camera(im_depth, meta_data)

        # compute icp
        if points_prev.shape[1] > 0:
            # sample model points
            X = points_prev[0,:]
            Y = points_prev[1,:]
            Z = points_prev[2,:]
            index = np.where(np.isfinite(X))[0]
            perm = np.random.permutation(np.arange(len(index)))
            num_model = min(1000, len(index))
            index = index[perm[:num_model]]
            model_points = np.zeros((num_model, 3), dtype=np.float64)
            model_points[:,0] = X[index]
            model_points[:,1] = Y[index]
            model_points[:,2] = Z[index]

            # sample template points
            X = points[0,:]
            Y = points[1,:]
            Z = points[2,:]
            index = np.where(np.isfinite(X))[0]
            perm = np.random.permutation(np.arange(len(index)))
            num_temp = min(1000, len(index))
            index = index[perm[:num_temp]]
            temp_points = np.zeros((num_temp, 3), dtype=np.float64)
            temp_points[:,0] = X[index]
            temp_points[:,1] = Y[index]
            temp_points[:,2] = Z[index]

            # initial transformation
            T = np.identity(4, dtype=np.float64)

            # perform icp
            Tr = icp_kernel.icp(model_points, temp_points, T, -1, 1)
            if len(transformations) > 0:
                Tr_prev = transformations[-1]
                transformations.append(np.dot(Tr, Tr_prev))
            else:
                transformations.append(Tr)

            """
            # apply the transformation to the template points
            temp_fit = np.dot(Tr[0:3, 0:3], np.transpose(temp_points)) + np.dot(Tr[0:3, 3].reshape((3,1)), np.ones((1, num_temp), dtype=np.float64))
            temp_fit = temp_fit.transpose()

            # compute camera position

            # show the current 3D points
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(temp_points[:,0], temp_points[:,1], temp_points[:,2], c='r', marker='o')
            # show the previous 3D points
            ax.scatter(model_points[:,0], model_points[:,1], model_points[:,2], c='g', marker='o')
            # draw camera position
            ax.scatter(0, 0, 0, c='g', marker='o')
            for j in range(len(transformations)):
                t = transformations[j]
                ax.scatter(t[0, 3], t[1, 3], t[2, 3], c='r', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            set_axes_equal(ax)
            plt.show()
            """

        points_prev = points
