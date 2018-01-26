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
from synthesize import synthesizer
from transforms3d.quaternions import quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

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

    height = 480
    width = 640
    fx = 572.41140
    fy = 573.57043
    px = 325.26110
    py = 242.04899
    zfar = 6.0
    znear = 0.25;
    factor= 1000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    root = '/home/yuxiang/Datasets/LINEMOD_SIXD/data/'

    synthesizer_ = synthesizer.PySynthesizer(args.cad_name, args.pose_name)
    synthesizer_.setup(width, height)

    # load data
    seq_id = 1
    frame_id = 1

    filename = root + '{:02d}/{:06d}-color.png'.format(seq_id, frame_id)
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    filename = root + '{:02d}/{:06d}-depth.png'.format(seq_id, frame_id)
    im_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    filename = root + '{:02d}/{:06d}-label.png'.format(seq_id, frame_id)
    labels_icp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    labels_icp = labels_icp.astype(np.int32)

    filename_mat = root + '{:02d}/{:06d}-meta.mat'.format(seq_id, frame_id)
    metadata = scipy.io.loadmat(filename_mat)

    num = len(metadata['cls_indexes'].flatten())
    poses = np.zeros((num, 7), dtype=np.float32)
    for i in xrange(num):
        R = metadata['poses'][:, :3, i]
        T = metadata['poses'][:, 3, i]
        poses[i, :4] = mat2quat(R)
        print R
        poses[i, 4:] = T
        print poses[i, :]

    boxes = metadata['boxes']
    rois = np.zeros((num, 6), dtype=np.float32)
    print rois.shape, rois
    print metadata['cls_indexes']
    rois[:,1] = metadata['cls_indexes'].flatten()
    rois[:,2] = boxes[:,0]
    rois[:,3] = boxes[:,1]
    rois[:,4] = boxes[:, 0] + boxes[:, 2]
    rois[:,5] = boxes[:, 1] + boxes[:, 3]

    poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
    poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
    error_threshold = 0.01
    synthesizer_.estimate_poses(labels_icp, im_depth, rois, poses, poses_new, poses_icp, fx, fy, px, py, znear, zfar, factor, error_threshold)

    RTs = np.zeros((3, 4, num), dtype=np.float32)
    for i in xrange(num):
        RTs[:, :3, i] = quat2mat(poses_icp[i, :4])
        RTs[:, 3, i] = poses_icp[i, 4:]
    metadata['poses_icp'] = RTs
    print metadata
    scipy.io.savemat(filename_mat, metadata, do_compression=True)


def vis_segmentations_vertmaps(im, im_depth, im_labels, im_labels_gt, colors,
  labels, labels_gt, rois, poses, poses_new, intrinsic_matrix, vertmap_gt, poses_gt, cls_indexes, num_classes, points):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 4, 1)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show gt class labels
    ax = fig.add_subplot(3, 4, 5)
    plt.imshow(im_labels_gt)
    ax.set_title('gt class labels')

    # show class label
    ax = fig.add_subplot(3, 4, 9)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    # show centers
    for i in xrange(rois.shape[0]):
        cx = (rois[i, 2] + rois[i, 4]) / 2
        cy = (rois[i, 3] + rois[i, 5]) / 2
        w = rois[i, 4] - rois[i, 2]
        h = rois[i, 5] - rois[i, 3]
        if not np.isinf(cx) and not np.isinf(cy):
            plt.plot(cx, cy, 'yo')

            # show boxes
            plt.gca().add_patch(
                plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                               edgecolor='g', linewidth=3))

    # show projection of the poses
    ax = fig.add_subplot(3, 4, 2, aspect='equal')
    plt.imshow(im)
    ax.invert_yaxis()
    for i in xrange(1, num_classes):
        index = np.where(labels_gt == i)
        if len(index[0]) > 0:
            # extract 3D points
            # num = len(index[0])
            x3d = np.ones((4, num), dtype=np.float32)
            x3d[0, :] = vertmap_gt[index[0], index[1], 0]
            x3d[1, :] = vertmap_gt[index[0], index[1], 1]
            x3d[2, :] = vertmap_gt[index[0], index[1], 2]

            # projection
            ind = np.where(cls_indexes == i)[0][0]
            RT = poses_gt[:, :, ind]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
            # plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[i], 255.0), alpha=0.05)
            plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[i], 255.0), s=10)

        ax.set_title('gt projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        ax = fig.add_subplot(3, 4, 3, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            index = np.where(labels_gt == cls)
            if len(index[0]) > 0:
                # extract 3D points
                # num = len(index[0])
                # x3d = np.ones((4, num), dtype=np.float32)
                # x3d[0, :] = vertmap_gt[index[0], index[1], 0]
                # x3d[1, :] = vertmap_gt[index[0], index[1], 1]
                # x3d[2, :] = vertmap_gt[index[0], index[1], 2]

                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                # plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)
                plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

        ax.set_title('projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        if cfg.TEST.POSE_REFINE:
            ax = fig.add_subplot(3, 4, 4, aspect='equal')
            plt.imshow(im)
            ax.invert_yaxis()
            for i in xrange(rois.shape[0]):
                cls = int(rois[i, 1])
                index = np.where(labels_gt == cls)
                if len(index[0]) > 0:
                    num = len(index[0])
                    # extract 3D points
                    x3d = np.ones((4, num), dtype=np.float32)
                    x3d[0, :] = vertmap_gt[index[0], index[1], 0]
                    x3d[1, :] = vertmap_gt[index[0], index[1], 1]
                    x3d[2, :] = vertmap_gt[index[0], index[1], 2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses_new[i, :4])
                    RT[:, 3] = poses_new[i, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

            ax.set_title('projection refined by ICP')
            ax.invert_yaxis()
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])

    plt.show()
