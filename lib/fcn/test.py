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
# from normals import gpu_normals
# from pose_estimation import ransac
from transforms3d.quaternions import quat2mat, mat2quat
# from kinect_fusion import kfusion
# from pose_refinement import refiner

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
    # mask the color image according to depth
    if cfg.EXP_DIR == 'rgbd_scene':
        I = np.where(im_depth == 0)
        im_orig[I[0], I[1], :] = 0

    processed_ims_rescale = []
    im_scale = cfg.TEST.SCALES_BASE[0]
    im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_rescale.append(im_rescale)

    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1

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

    # meta data
    K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # normals
    '''
    depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
    nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
    im_normal = 127.5 * nmap + 127.5
    im_normal = im_normal.astype(np.uint8)
    im_normal = im_normal[:, :, (2, 1, 0)]
    im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

    im_orig = im_normal.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_normal = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_normal.append(im)
    '''

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)
    # blob_normal = im_list_to_blob(processed_ims_normal, 3)
    blob_normal = []

    return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, extents, num_classes):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

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
    # mdata[18:30] = pose_world2live.flatten()
    # mdata[30:42] = pose_live2world.flatten()
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
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    meta_data_blob[0,0,0,:] = mdata

    # use a fake label blob of ones
    height = im_depth.shape[0]
    width = im_depth.shape[1]
    label_blob = np.ones((1, height, width, num_classes), dtype=np.float32)

    pose_blob = np.zeros((1, 13), dtype=np.float32)
    vertex_target_blob = np.zeros((1, height, width, 2*num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 2*num_classes), dtype=np.float32)

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.POSE_REG:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.NETWORK == 'FCN8VGG':
        labels_2d, probs = sess.run([net.label_2d, net.prob], feed_dict=feed_dict)
    else:
        if cfg.TEST.VERTEX_REG:
            labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                          net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_pred')], feed_dict=feed_dict)
            # combine poses
            num = rois.shape[0]
            poses = poses_init
            for i in xrange(num):
                class_id = int(rois[i, 1])
                poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            print poses

            # average rois
            n = 9
            rois_new = np.zeros((num/n, 6), dtype=np.float32);
            poses_new = np.zeros((num/n, 7), dtype=np.float32);
            for i in xrange(num/n):
                rois_new[i, :] = np.mean(rois[n*i:n*(i+1), :], axis=0)
                poses_new[i, :] = np.mean(poses[n*i:n*(i+1), :], axis=0)

        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')], feed_dict=feed_dict)
            vertex_pred = []
            rois = []
            poses = []

    return labels_2d[0,:,:].astype(np.uint8), probs[0,:,:,:], vertex_pred, rois_new, poses_new


def im_segment(sess, net, im, im_depth, state, weights, points, meta_data, voxelizer, pose_world2live, pose_live2world):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)

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
    im_normal_blob = im_normal_blob.reshape((num_steps, ims_per_batch, height_blob, width_blob, -1))

    label_blob = label_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    depth_blob = depth_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    meta_data_blob = meta_data_blob.reshape((num_steps, ims_per_batch, 1, 1, -1))

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.state: state, net.weights: weights, net.depth: depth_blob, \
                     net.meta_data: meta_data_blob, net.points: points, net.keep_prob: 1.0}
    else:
        feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.state: state, net.weights: weights, net.depth: depth_blob, \
                     net.meta_data: meta_data_blob, net.points: points, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    labels_pred_2d, probs, state, weights, points = sess.run([net.get_output('labels_pred_2d'), net.get_output('probs'), \
        net.get_output('output_state'), net.get_output('output_weights'), net.get_output('output_points')], feed_dict=feed_dict)

    labels_2d = labels_pred_2d[0]

    return labels_2d[0,:,:].astype(np.uint8), probs[0][0,:,:,:], state, weights, points


def vis_segmentations(im, im_depth, labels, labels_gt, colors):
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

    points = points[0,:,:,:].reshape((-1, 3))

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    index = np.where(np.isfinite(X))[0]
    perm = np.random.permutation(np.arange(len(index)))
    num = min(10000, len(index))
    index = index[perm[:num]]
    ax.scatter(X[index], Y[index], Z[index], c='r', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    '''
    plt.show()

##################
# test video
##################
def test_net(sess, net, imdb, weights_filename, rig_filename, is_kfusion):

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
    voxelizer.setup(-3, -3, -3, 3, 3, 4)
    # voxelizer.setup(-2, -2, -2, 2, 2, 2)

    # kinect fusion
    if is_kfusion:
        KF = kfusion.PyKinectFusion(rig_filename)

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
    else:
        perm = xrange(num_images)

    video_index = ''
    have_prediction = False
    for i in perm:
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        height = rgba.shape[0]
        width = rgba.shape[1]

        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            have_prediction = False
            state = np.zeros((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            weights = np.ones((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
            points = np.zeros((1, height, width, 3), dtype=np.float32)
        else:
            if video_index != image_index[:pos]:
                have_prediction = False
                video_index = image_index[:pos]
                state = np.zeros((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                weights = np.ones((1, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
                points = np.zeros((1, height, width, 3), dtype=np.float32)
                print 'start video {}'.format(video_index)

        # read color image
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # read depth image
        im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        # backprojection for the first frame
        if not have_prediction:    
            if is_kfusion:
                # KF.set_voxel_grid(-3, -3, -3, 6, 6, 7)
                KF.set_voxel_grid(voxelizer.min_x, voxelizer.min_y, voxelizer.min_z, voxelizer.max_x-voxelizer.min_x, voxelizer.max_y-voxelizer.min_y, voxelizer.max_z-voxelizer.min_z)
                # identity transformation
                RT_world = np.zeros((3,4), dtype=np.float32)
                RT_world[0, 0] = 1
                RT_world[1, 1] = 1
                RT_world[2, 2] = 1
            else:
                # store the RT for the first frame
                RT_world = meta_data['rotation_translation_matrix']

        # run kinect fusion
        if is_kfusion:
            im_rgb = np.copy(im)
            im_rgb[:, :, 0] = im[:, :, 2]
            im_rgb[:, :, 2] = im[:, :, 0]
            KF.feed_data(im_depth, im_rgb, im.shape[1], im.shape[0], float(meta_data['factor_depth']))
            KF.back_project();
            if have_prediction:
                pose_world2live, pose_live2world = KF.solve_pose()
                RT_live = pose_world2live
            else:
                RT_live = RT_world
        else:
            # compute camera poses
            RT_live = meta_data['rotation_translation_matrix']

        pose_world2live = se3_mul(RT_live, se3_inverse(RT_world))
        pose_live2world = se3_inverse(pose_world2live)

        _t['im_segment'].tic()
        labels, probs, state, weights, points = im_segment(sess, net, im, im_depth, state, weights, points, meta_data, voxelizer, pose_world2live, pose_live2world)
        _t['im_segment'].toc()
        # time.sleep(3)

        _t['misc'].tic()
        labels = unpad_im(labels, 16)

        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        if is_kfusion:
            labels_kfusion = np.zeros((height, width), dtype=np.int32)
            if probs.shape[2] < 10:
                probs_new = np.zeros((probs.shape[0], probs.shape[1], 10), dtype=np.float32)
                probs_new[:,:,:imdb.num_classes] = probs
                probs = probs_new
            KF.feed_label(im_label, probs, colors)
            KF.fuse_depth()
            labels_kfusion = KF.extract_surface(labels_kfusion)
            im_label_kfusion = imdb.labels_to_image(im, labels_kfusion)
            KF.render()
            filename = os.path.join(output_dir, 'images', '{:04d}'.format(i))
            KF.draw(filename, 0)
        have_prediction = True

        # compute the delta transformation between frames
        RT_world = RT_live

        if is_kfusion:
            seg = {'labels': labels_kfusion}
        else:
            seg = {'labels': labels}
        segmentations[i] = seg

        _t['misc'].toc()

        if cfg.TEST.VISUALIZE:
            # read label image
            labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)
            if len(labels_gt.shape) == 2:
                im_label_gt = imdb.labels_to_image(im, labels_gt)
            else:
                im_label_gt = np.copy(labels_gt[:,:,:3])
                im_label_gt[:,:,0] = labels_gt[:,:,2]
                im_label_gt[:,:,2] = labels_gt[:,:,0]
            vis_segmentations(im, im_depth, im_label, im_label_gt, imdb._class_colors)

        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

    if is_kfusion:
        KF.draw(filename, 1)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)

# compute the voting label image in 2D
def _vote_centers(im_label, cls_indexes, centers, num_classes):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)

    center = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        if len(x) > 0:
            ind = np.where(cls_indexes == i)[0]
            center[0] = centers[ind, 0]
            center[1] = centers[ind, 1]
            R = np.tile(center, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            vertex_targets[y, x, 0] = R[0,:]
            vertex_targets[y, x, 1] = R[1,:]

    return vertex_targets


# extract vertmap for vertex predication
def _extract_vertmap(im_label, vertex_pred, extents, num_classes):
    height = im_label.shape[0]
    width = im_label.shape[1]
    vertmap = np.zeros((height, width, 2), dtype=np.float32)
    # centermap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(im_label == i)
        if len(I[0]) > 0:
            start = 2 * i
            end = 2 * i + 2
            vertmap[I[0], I[1], :] = vertex_pred[0, I[0], I[1], start:end]

            # start = 2 * i
            # end = 2 * i + 2
            # centermap[I[0], I[1], :2] = vertex_pred[0, I[0], I[1], start:end]

    return vertmap
    #return _unscale_vertmap(vertmap, im_label, extents, num_classes)
    #return vertmap, centermap  


def scale_vertmap(vertmap):
    vmin = vertmap.min()
    vmax = vertmap.max()
    if vmax - vmin > 0:
        a = 1.0 / (vmax - vmin)
        b = -1.0 * vmin / (vmax - vmin)
    else:
        a = 0
        b = 0
    return a * vertmap + b


def _unscale_vertmap(vertmap, labels, extents, num_classes):
    for k in range(1, num_classes):
        index = np.where(labels == k)
        for i in range(3):
            vmin = -extents[k, i] / 2
            vmax = extents[k, i] / 2
            a = 1.0 / (vmax - vmin)
            b = -1.0 * vmin / (vmax - vmin)
            vertmap[index[0], index[1], i] = (vertmap[index[0], index[1], i] - b) / a
    return vertmap


def vis_segmentations_vertmaps(im, im_depth, im_labels, im_labels_gt, colors, center_gt, center, labels, labels_gt, rois, intrinsic_matrix, vertmap_gt, poses, num_classes):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(241)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show gt class labels
    ax = fig.add_subplot(242)
    plt.imshow(im_labels_gt)
    ax.set_title('gt class labels')

    # show depth
    # ax = fig.add_subplot(245)
    # plt.imshow(im_depth)
    # ax.set_title('input depth')

    # show gt vertex map
    ax = fig.add_subplot(243)
    plt.imshow(center_gt[:,:,0])
    ax.set_title('gt centers x')

    ax = fig.add_subplot(244)
    plt.imshow(center_gt[:,:,1])
    ax.set_title('gt centers y')

    # show class label
    ax = fig.add_subplot(246)
    plt.imshow(im_labels)
    ax.set_title('class labels')

    # show centers
    centers_x = (rois[:, 2] + rois[:, 4]) / 2
    centers_y = (rois[:, 3] + rois[:, 5]) / 2
    plt.plot(centers_x, centers_y, 'ro')

    # show boxes
    for i in xrange(rois.shape[0]):
        plt.gca().add_patch(
            plt.Rectangle((rois[i, 2], rois[i, 3]), rois[i, 4] - rois[i, 2],
                          rois[i, 5] - rois[i, 3], fill=False,
                          edgecolor='g', linewidth=3)
            )

    # show vertex map
    ax = fig.add_subplot(247)
    plt.imshow(center[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(248)
    plt.imshow(center[:,:,1])
    ax.set_title('centers y')

    # show projection of the poses
    if cfg.TEST.POSE_REG:
        ax = fig.add_subplot(245, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            index = np.where(labels_gt == cls)
            if len(index[0]) > 0:
                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:]

                print RT

                num = len(index[0])
                # extract 3D points
                x3d = np.ones((4, num), dtype=np.float32)
                x3d[0, :] = vertmap_gt[index[0], index[1], 0]
                x3d[1, :] = vertmap_gt[index[0], index[1], 1]
                x3d[2, :] = vertmap_gt[index[0], index[1], 2]

                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
          
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[i], 255.0), alpha=0.05)
        ax.set_title('projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

    plt.show()


###################
# test single frame
###################
def test_net_single_frame(sess, net, imdb, weights_filename, model_filename, is_refine):

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
    voxelizer.setup(-3, -3, -3, 3, 3, 4)

    # kinect fusion
    if is_refine:
        RF = refiner.PyRefiner(model_filename)

    # pose estimation
    if cfg.TEST.RANSAC:
        RANSAC = ransac.PyRansac3D()

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
        # perm = xrange(0, num_images, 200)
    else:
        perm = xrange(num_images)

    video_index = ''
    for i in perm:

        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
        else:
            if video_index != image_index[:pos]:
                video_index = image_index[:pos]

        # read color image
        rgba = pad_im(cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
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

        _t['im_segment'].tic()
        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, imdb._extents, imdb.num_classes)
        if cfg.TEST.VERTEX_REG:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            # centers = RANSAC.estimate_center(probs, vertex_pred[0,:,:,:])
            # print centers
            if cfg.TEST.RANSAC:
                # pose estimation using RANSAC
                fx = meta_data['intrinsic_matrix'][0, 0]
                fy = meta_data['intrinsic_matrix'][1, 1]
                px = meta_data['intrinsic_matrix'][0, 2]
                py = meta_data['intrinsic_matrix'][1, 2]
                depth_factor = meta_data['factor_depth'][0, 0]
                poses = RANSAC.estimate_pose(im_depth, probs, vertex_pred[0,:,:,:] / cfg.TRAIN.VERTEX_W, imdb._extents, fx, fy, px, py, depth_factor)

                # print gt poses
                # cls_indexes = meta_data['cls_indexes']
                # poses_gt = meta_data['poses']
                # for j in xrange(len(cls_indexes)):
                #    print 'object {}'.format(cls_indexes[j])
                #    print poses_gt[:,:,j]

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels = unpad_im(labels, 16)
        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        # run kinect fusion
        if is_refine:
            poses_gt = meta_data['poses']
            cls_indexes = meta_data['cls_indexes']
            num = poses_gt.shape[2]
            qt = np.zeros((num, 13), dtype=np.float32)
            for j in xrange(num):
                R = poses_gt[:, :3, j]
                T = poses_gt[:, 3, j]

                qt[j, 0] = 0
                qt[j, 1] = cls_indexes[j]
                # qt[j, 2:6] = roidb[i]['boxes'][j, :]
                qt[j, 6:10] = mat2quat(R)
                qt[j, 10:] = T

            fx = meta_data['intrinsic_matrix'][0, 0]
            fy = meta_data['intrinsic_matrix'][1, 1]
            px = meta_data['intrinsic_matrix'][0, 2]
            py = meta_data['intrinsic_matrix'][1, 2]
            poses_new = np.zeros_like(poses)
            RF.render(im, labels, rois, qt, poses, fx, fy, px, py, imdb.num_classes, imdb._extents, poses_new, 0);
            
        seg = {'labels': labels}
        segmentations[i] = seg

        _t['misc'].toc()

        print 'im_segment {}: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(video_index, i + 1, num_images, _t['im_segment'].diff, _t['misc'].diff)

        if cfg.TEST.VISUALIZE:
            if cfg.TEST.VERTEX_REG:
                centers_gt = _vote_centers(labels_gt, meta_data['cls_indexes'], meta_data['center'], imdb.num_classes)
                print 'visualization'
                vis_segmentations_vertmaps(im, im_depth, im_label, im_label_gt, imdb._class_colors, \
                    centers_gt, vertmap, labels, labels_gt, rois, meta_data['intrinsic_matrix'], meta_data['vertmap'], poses, imdb.num_classes)
            else:
                vis_segmentations(im, im_depth, im_label, im_label_gt, imdb._class_colors)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
