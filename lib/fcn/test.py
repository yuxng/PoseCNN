# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an imdb (image database)."""

from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
from utils.blob import im_list_to_blob, pad_im
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import scipy.io
from kinect_fusion import kfusion
import time

def _get_image_blob(im, im_depth):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    # RGB
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1
    im_scale = cfg.TEST.SCALES_BASE[0]

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # depth
    im_orig = im_depth.astype(np.float32, copy=True)
    im_orig = im_orig / im_orig.max() * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # backprojection
    points = voxelizer.backproject(im_depth, meta_data)
    voxelizer.voxelized = False
    grid_indexes = voxelizer.voxelize(points)

    # construct the meta data
    """
    format of the meta_data
    projection matrix: meta_data[0 ~ 11]
    camera center: meta_data[12, 13, 14]
    voxel step size: meta_data[15, 16, 17]
    voxel min value: meta_data[18, 19, 20]
    backprojection matrix: meta_data[21 ~ 32]
    """
    P = np.matrix(meta_data['projection_matrix'])
    Pinv = np.linalg.pinv(P)
    mdata = np.zeros(33, dtype=np.float32)
    mdata[0:12] = P.flatten()
    mdata[12:15] = meta_data['camera_location']
    mdata[15] = voxelizer.step_x
    mdata[16] = voxelizer.step_y
    mdata[17] = voxelizer.step_z
    mdata[18] = voxelizer.min_x
    mdata[19] = voxelizer.min_y
    mdata[20] = voxelizer.min_z
    mdata[21:33] = Pinv.flatten()

    # construct blobs
    height = im_depth.shape[0]
    width = im_depth.shape[1]
    depth_blob = np.zeros((1, height, width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((1, 1, 1, 33), dtype=np.float32)
    depth_blob[0,:,:,0] = depth
    meta_data_blob[0,0,0,:] = mdata
    # use a fake label blob
    label_blob = np.zeros((1, height, width, voxelizer.num_classes), dtype=np.float32)

    # forward pass
    feed_dict = {net.data: im_depth_blob, net.label: label_blob, net.depth: depth_blob, net.meta_data: meta_data_blob}
    output = sess.run([net.get_output('label')], feed_dict=feed_dict)
    labels = output[0]

    return labels[0,:,:,0], points


def im_segment(sess, net, im, im_depth, state, label_3d, meta_data, voxelizer, pose_world2live, pose_live2world):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # construct the meta data
    """
    format of the meta_data
    intrinsic matrix: meta_data[0 ~ 8]
    inverse intrinsic matrix: meta_data[9 ~ 17]
    pose_world2live: meta_data[18 ~ 29]
    pose_live2world: meta_data[30 ~ 41]
    voxel step size: meta_data[42, 43, 44]
    voxel min value: meta_data[45, 46, 47]
    """
    K = np.matrix(meta_data['intrinsic_matrix'])
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    mdata[18:30] = pose_world2live.flatten()
    mdata[30:42] = pose_live2world.flatten()
    mdata[42] = voxelizer.step_x
    mdata[43] = voxelizer.step_y
    mdata[44] = voxelizer.step_z
    mdata[45] = voxelizer.min_x
    mdata[46] = voxelizer.min_y
    mdata[47] = voxelizer.min_z
    if cfg.FLIP_X:
        mdata[0] = -1 * mdata[0]
        mdata[9] = -1 * mdata[9]
        mdata[11] = -1 * mdata[11]

    # construct blobs
    height = im_depth.shape[0]
    width = im_depth.shape[1]
    depth_blob = np.zeros((1, height, width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    depth_blob[0,:,:,0] = depth
    meta_data_blob[0,0,0,:] = mdata
    # use a fake label blob of 1s
    label_blob = np.ones((1, height, width, voxelizer.num_classes), dtype=np.float32)

    # reshape the blobs
    num_steps = 1
    ims_per_batch = 1
    height_blob = im_blob.shape[1]
    width_blob = im_blob.shape[2]
    im_blob = im_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))
    im_depth_blob = im_depth_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))

    label_blob = label_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    depth_blob = depth_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    meta_data_blob = meta_data_blob.reshape((num_steps, ims_per_batch, 1, 1, -1))
    im_rgbd_blob = np.concatenate((im_blob, im_depth_blob), axis=4)

    # forward pass
    feed_dict = {net.data: im_rgbd_blob, net.gt_label_2d: label_blob, net.state: state, net.depth: depth_blob, \
                 net.meta_data: meta_data_blob, net.gt_label_3d: label_3d}
    labels_pred_2d, labels_pred_3d, state, label_3d = sess.run([net.get_output('labels_pred_2d'), net.get_output('labels_pred_3d'), \
        net.get_output('output_state'),  net.get_output('output_label_3d')], feed_dict=feed_dict)

    labels_2d = labels_pred_2d[0]
    labels_3d = labels_pred_3d[0]

    return labels_2d[0,:,:,0], labels_3d[0,:,:,:].astype(np.int32), state, label_3d


def vis_segmentations(im, im_depth, labels, labels_gt, points):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(221)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(222)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(223)
    plt.imshow(labels)
    ax.set_title('class labels')

    ax = fig.add_subplot(224)
    plt.imshow(labels_gt)
    ax.set_title('gt class labels')

    # show the 3D points
    '''
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(224, projection='3d')
    ax.set_aspect('equal')
    X = points[0,:]
    Y = points[1,:]
    Z = points[2,:]
    index = np.where(np.isfinite(X))[0]
    perm = np.random.permutation(np.arange(len(index)))
    num = min(10000, len(index))
    index = index[perm[:num]]
    ax.scatter(X[index], Y[index], Z[index], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    '''
    plt.show()

##################
# test video
##################
def test_net(sess, net, imdb, weights_filename, rig_filename):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    print imdb.name
    if os.path.exists(seg_file):
        with open(seg_file, 'rb') as fid:
            segmentations = cPickle.load(fid)
        imdb.evaluate_segmentations(segmentations, output_dir)
        return

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)

    # kinect fusion
    if cfg.TEST.KINECT_FUSION:
        KF = kfusion.PyKinectFusion(rig_filename)

    # construct colors
    colors = np.zeros((3, imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[0, i] = 255 * imdb._class_colors[i][0]
        colors[1, i] = 255 * imdb._class_colors[i][1]
        colors[2, i] = 255 * imdb._class_colors[i][2]

    # perm = np.random.permutation(np.arange(num_images))

    video_index = ''
    video_count = 0
    have_prediction = False
    for i in xrange(num_images):
    # for i in perm:
        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            video_count = 0
            voxelizer.reset()
            have_prediction = False
            state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
        else:
            if video_index != image_index[:pos]:
                voxelizer.reset()
                have_prediction = False
                video_count = 0
                video_index = image_index[:pos]
                state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
                print 'start video {}'.format(video_index)
            else:
                if video_count % 1000 == 0:
                    voxelizer.reset()
                    have_prediction = False
                    state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                    label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
                    print 'restart video {}'.format(video_index)
        video_count += 1

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = rgba[:,:,:3]
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 255
        else:
            im = rgba

        # read depth image
        im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        # read label image
        im_label_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # backprojection for the first frame
        points = voxelizer.backproject_camera(im_depth, meta_data)
        if not have_prediction:    
            voxelizer.voxelize(points)
            if cfg.TEST.KINECT_FUSION:
                KF.set_voxel_grid(voxelizer.min_x, voxelizer.min_y, voxelizer.min_z, voxelizer.max_x-voxelizer.min_x, voxelizer.max_y-voxelizer.min_y, voxelizer.max_z-voxelizer.min_z)
            else:
                # store the RT for the first frame
                RT_world = meta_data['rotation_translation_matrix']

        # run kinect fusion
        if cfg.TEST.KINECT_FUSION:
            KF.feed_data(im_depth, rgba, im.shape[1], im.shape[0])
            KF.back_project();
            if have_prediction:
                pose_world2live, pose_live2world = KF.solve_pose()
            else:
                pose_world2live = np.zeros((3,4), dtype=np.float32)
                pose_world2live[0, 0] = 1
                pose_world2live[1, 1] = 1
                pose_world2live[2, 2] = 1
                pose_live2world = pose_world2live
        else:
            # compute camera poses
            RT_live = meta_data['rotation_translation_matrix']
            pose_world2live = se3_mul(RT_live, se3_inverse(RT_world))
            pose_live2world = se3_inverse(pose_world2live)

        print pose_world2live
        print pose_live2world

        _t['im_segment'].tic()
        labels, labels_voxel, state, label_3d = im_segment(sess, net, im, im_depth, state, label_3d, meta_data, voxelizer, pose_world2live, pose_live2world)
        _t['im_segment'].toc()
        # time.sleep(3)

        _t['misc'].tic()
        seg = {'labels': labels}
        segmentations[i] = seg

        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        if cfg.TEST.KINECT_FUSION:
            KF.fuse_depth()
            KF.extract_surface()
            KF.render()
            KF.feed_label(im_label, labels_voxel, colors)
            KF.draw()
        have_prediction = True

        # show voxel labels
        # voxelizer.draw(labels_voxel, imdb._class_colors)

        _t['misc'].toc()

        vis_segmentations(im, im_depth, im_label, im_label_gt, points)
        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)

###################
# test single frame
###################
def test_net_single_frame(sess, net, imdb, weights_filename):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    print imdb.name
    if os.path.exists(seg_file):
        with open(seg_file, 'rb') as fid:
            segmentations = cPickle.load(fid)
        imdb.evaluate_segmentations(segmentations, output_dir)
        return

    """Test a FCN on an image database."""
    num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)

    # perm = np.random.permutation(np.arange(num_images))

    for i in xrange(num_images):
    # for i in perm:

        rgba = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        _t['im_segment'].tic()
        labels, points = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer)
        _t['im_segment'].toc()

        _t['misc'].tic()
        seg = {'labels': labels}
        segmentations[i] = seg
        _t['misc'].toc()

        # vis_segmentations(im, im_depth, labels, points)
        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
