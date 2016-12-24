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
from utils.blob import im_list_to_blob, pad_im, unpad_im
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import scipy.io
import time
from normals import gpu_normals
#from kinect_fusion import kfusion

def _get_image_blob(im, im_depth, meta_data):
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

    # meta data
    K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # normals
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
    nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
    im_normal = 127.5 * nmap + 127.5
    im_normal = im_normal.astype(np.uint8)
    im_normal = im_normal[:, :, (2, 1, 0)]
    im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

    im_orig = im_normal.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, pose_world2live, pose_live2world):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)
    im_rgbd_blob = np.concatenate((im_blob, im_depth_blob), axis=3)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])

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
    # use a fake label blob of ones
    label_blob = np.ones((1, height, width, voxelizer.num_classes), dtype=np.float32)
    label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, voxelizer.num_classes), dtype=np.float32)

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_rgbd_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_depth_blob
    feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.depth: depth_blob, \
                 net.meta_data: meta_data_blob, net.gt_label_3d: label_3d}

    # labels_2d, labels_3d = sess.run([net.get_output('label_2d'), net.get_output('label_3d')], feed_dict=feed_dict)
    # return labels_2d[0,:,:,0], labels_3d[0,:,:,:].astype(np.int32)

    output = sess.run([net.get_output('label_2d')], feed_dict=feed_dict)
    labels_2d = output[0]
    labels_3d = np.zeros((cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE), dtype=np.int32)
    return labels_2d[0,:,:], labels_3d


def im_segment(sess, net, im, im_depth, state, label_3d, meta_data, voxelizer, pose_world2live, pose_live2world):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])

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
    if cfg.INPUT == 'RGBD':
        data_blob = im_rgbd_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_depth_blob
    feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.state: state, net.depth: depth_blob, \
                 net.meta_data: meta_data_blob, net.gt_label_3d: label_3d}
    labels_pred_2d, labels_pred_3d, state, label_3d = sess.run([net.get_output('labels_pred_2d'), net.get_output('labels_pred_3d'), \
        net.get_output('output_state'),  net.get_output('output_label_3d')], feed_dict=feed_dict)

    labels_2d = labels_pred_2d[0]
    labels_3d = labels_pred_3d[0]

    return labels_2d[0,:,:,0].astype(np.int32), labels_3d[0,:,:,:].astype(np.int32), state, label_3d


def vis_segmentations(im, im_depth, labels, labels_gt, labels_voxel, colors, voxelizer):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(221)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(222, projection='3d')
    voxelizer.draw(labels_voxel, colors, ax)
    # plt.imshow(im_depth)
    # ax.set_title('input depth')

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
    colors[:,0] = 255

    # perm = np.random.permutation(np.arange(num_images))

    video_index = ''
    video_count = 0
    have_prediction = False
    restart = False
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
            restart = False
            state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
        else:
            if video_index != image_index[:pos]:
                voxelizer.reset()
                have_prediction = False
                restart = False
                video_count = 0
                video_index = image_index[:pos]
                state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
                print 'start video {}'.format(video_index)
            else:
                if restart:
                    voxelizer.reset()
                    have_prediction = False
                    restart = False
                    state = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                    label_3d = np.zeros((1, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, cfg.TEST.GRID_SIZE, imdb.num_classes), dtype=np.float32)
                    print 'restart video {}'.format(video_index)
        video_count += 1

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
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
        labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if len(labels_gt.shape) == 2:
            im_label_gt = imdb.labels_to_image(im, labels_gt)
        else:
            im_label_gt = np.copy(labels_gt[:,:,:3])
            im_label_gt[:,:,0] = labels_gt[:,:,2]
            im_label_gt[:,:,2] = labels_gt[:,:,0]

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
            KF.feed_data(im_depth, im, im.shape[1], im.shape[0], float(meta_data['factor_depth']))
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

        # check if points outside voxel space
        flag = voxelizer.check_points(points, pose_live2world)
        if not flag:
            print 'points outside voxel space, restart from next frame'
            restart = True

        _t['im_segment'].tic()
        labels, labels_voxel, state, label_3d = im_segment(sess, net, im, im_depth, state, label_3d, meta_data, voxelizer, pose_world2live, pose_live2world)
        _t['im_segment'].toc()
        # time.sleep(3)

        _t['misc'].tic()
        labels = unpad_im(labels, 16)
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

        vis_segmentations(im, im_depth, im_label, im_label_gt, labels_voxel, imdb._class_colors, voxelizer)
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

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
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
        labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if len(labels_gt.shape) == 2:
            im_label_gt = imdb.labels_to_image(im, labels_gt)
        else:
            im_label_gt = np.copy(labels_gt[:,:,:3])
            im_label_gt[:,:,0] = labels_gt[:,:,2]
            im_label_gt[:,:,2] = labels_gt[:,:,0]

        # backprojection
        points = voxelizer.backproject_camera(im_depth, meta_data) 
        voxelizer.voxelize(points)
        pose_world2live = np.zeros((3,4), dtype=np.float32)
        pose_world2live[0, 0] = 1
        pose_world2live[1, 1] = 1
        pose_world2live[2, 2] = 1
        pose_live2world = pose_world2live

        _t['im_segment'].tic()
        labels, labels_voxel = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, pose_world2live, pose_live2world)
        _t['im_segment'].toc()

        _t['misc'].tic()
        labels = unpad_im(labels, 16)
        seg = {'labels': labels}
        segmentations[i] = seg

        # build the label image
        im_label = imdb.labels_to_image(im, labels)
        _t['misc'].toc()

        # vis_segmentations(im, im_depth, im_label, im_label_gt, labels_voxel, imdb._class_colors, voxelizer)
        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
