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
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
from utils.pose_error import *
from utils.bbox_transform import clip_boxes, bbox_transform_inv
from utils.nms import nms
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import time
from transforms3d.quaternions import quat2mat, mat2quat
import scipy.io
from scipy.optimize import minimize
from normals import gpu_normals

# from synthesize import synthesizer
# from pose_estimation import ransac
# from kinect_fusion import kfusion
# from pose_refinement import refiner
# from mpl_toolkits.mplot3d import Axes3D

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
    # im_orig = im_orig / im_orig.max() * 255
    im_orig = np.clip(im_orig / 2000.0, 0, 1) * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    if cfg.INPUT == 'NORMAL':
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

        processed_ims_normal = []
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_normal.append(im)
        blob_normal = im_list_to_blob(processed_ims_normal, 3)
    else:
        blob_normal = []

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, extents, points, symmetry, num_classes):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)
    im_scale = im_scale_factors[0]
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
    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1
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
    height = int(im_depth.shape[0] * im_scale)
    width = int(im_depth.shape[1] * im_scale)
    label_blob = np.ones((1, height, width), dtype=np.int32)

    pose_blob = np.zeros((1, 13), dtype=np.float32)
    vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

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
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.NETWORK == 'FCN8VGG':
        labels_2d, probs = sess.run([net.label_2d, net.prob], feed_dict=feed_dict)
    else:
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

                # non-maximum suppression
                keep = nms(rois, 0.5)
                rois = rois[keep, :]
                poses_init = poses_init[keep, :]
                poses_pred = poses_pred[keep, :]
                print keep
                print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                print rois
                print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        elif cfg.TEST.VERTEX_REG_3D:
            labels_2d, probs, vertex_pred = \
                sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred')])
            rois = []
            poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

    return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses


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

    return labels_2d[0,:,:].astype(np.int32), probs[0][0,:,:,:], state, weights, points


def vis_segmentations(im, im_depth, labels, labels_gt, colors):
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
def _vote_centers(im_label, cls_indexes, centers, poses, num_classes, extents):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)

    center = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        I = np.where(im_label == i)
        ind = np.where(cls_indexes == i)[0]
        if len(x) > 0 and len(ind) > 0:
            center[0] = centers[ind, 0]
            center[1] = centers[ind, 1]
            z = poses[2, 3, ind]
            R = np.tile(center, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            vertex_targets[y, x, 0] = R[0,:]
            vertex_targets[y, x, 1] = R[1,:]
            vertex_targets[y, x, 2] = z

    return vertex_targets


# extract vertmap for vertex predication
def _extract_vertmap(im_label, vertex_pred, extents, num_classes):
    height = im_label.shape[0]
    width = im_label.shape[1]
    vertmap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(im_label == i)
        if len(I[0]) > 0:
            start = 3 * i
            end = 3 * i + 3
            vertmap[I[0], I[1], :] = vertex_pred[I[0], I[1], start:end]
    vertmap[:, :, 2] = np.exp(vertmap[:, :, 2])
    return vertmap

# compute the voting label image in 2D
def _generate_vertex_targets(im_label, num_classes, vertmap, extents):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        I = np.where(im_label == i)
        if len(x) > 0:
            vertex_targets[y, x, :] = _scale_vertmap(vertmap, I, extents[i, :])

    return vertex_targets


def _scale_vertmap(vertmap, index, extents):
    for i in range(3):
        vmin = -extents[i] / 2
        vmax = extents[i] / 2
        if vmax - vmin > 0:
            a = 1.0 / (vmax - vmin)
            b = -1.0 * vmin / (vmax - vmin)
        else:
            a = 0
            b = 0
        vertmap[index[0], index[1], i] = a * vertmap[index[0], index[1], i] + b
    return vertmap[index[0], index[1], :]


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


def vis_segmentations_vertmaps(im, im_depth, im_labels, im_labels_gt, colors, center_map_gt, center_map, 
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

    # show gt vertex map
    ax = fig.add_subplot(3, 4, 6)
    plt.imshow(center_map_gt[:,:,0])
    ax.set_title('gt centers x')

    ax = fig.add_subplot(3, 4, 7)
    plt.imshow(center_map_gt[:,:,1])
    ax.set_title('gt centers y')
    
    ax = fig.add_subplot(3, 4, 8)
    plt.imshow(center_map_gt[:,:,2])
    index = np.where(center_map_gt[:,:,2] > 0)
    if len(index[0]) > 0:
        z_gt = center_map_gt[index[0][0], index[1][0], 2]
        ax.set_title('gt centers z: {:6f}'.format(z_gt))
    else:
        ax.set_title('gt centers z')

    # show class label
    ax = fig.add_subplot(3, 4, 9)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    if cfg.TEST.VERTEX_REG_2D:
        # show centers
        for i in xrange(rois.shape[0]):
            if rois[i, 1] == 0:
                continue
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
        
    # show vertex map
    ax = fig.add_subplot(3, 4, 10)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 4, 11)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 4, 12)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z: {:6f}'.format(poses[0, 6]))

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 4, 2, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(1, num_classes):
            index = np.where(labels_gt == i)
            if len(index[0]) > 0:
                # extract 3D points
                if len(vertmap_gt.shape) == 3:
                    num = len(index[0])
                    x3d = np.ones((4, num), dtype=np.float32)
                    x3d[0, :] = vertmap_gt[index[0], index[1], 0]
                    x3d[1, :] = vertmap_gt[index[0], index[1], 1]
                    x3d[2, :] = vertmap_gt[index[0], index[1], 2]
                else:
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[i,:,0]
                    x3d[1, :] = points[i,:,1]
                    x3d[2, :] = points[i,:,2]

                # projection
                ind = np.where(cls_indexes == i)[0][0]
                RT = poses_gt[:, :, ind]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[i], 255.0), alpha=0.05)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[i], 255.0), s=10)

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
            if len(index[0]) > 0 and cls > 0:
                # extract 3D points
                if len(vertmap_gt.shape) == 3:
                    num = len(index[0])
                    x3d = np.ones((4, num), dtype=np.float32)
                    x3d[0, :] = vertmap_gt[index[0], index[1], 0]
                    x3d[1, :] = vertmap_gt[index[0], index[1], 1]
                    x3d[2, :] = vertmap_gt[index[0], index[1], 2]
                else:
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
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

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
                if len(index[0]) > 0 and cls > 0:
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


def vis_segmentations_vertmaps_detection(im, im_depth, im_labels, colors, center_map, 
  labels, rois, poses, poses_new, intrinsic_matrix, num_classes, classes, points):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 3, 1)
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(3, 3, 2)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(3, 3, 3)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    if cfg.TEST.VERTEX_REG_2D:
        # show centers
        for i in xrange(rois.shape[0]):
            if rois[i, 1] == 0:
                continue
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
        
    # show vertex map
    ax = fig.add_subplot(3, 3, 4)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 3, 5)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 3, 6)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z')

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 3, 7, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0:
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                print classes[cls]
                print RT
                print '\n'
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.5)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

        ax.set_title('projection of model points')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        if cfg.TEST.POSE_REFINE:
            ax = fig.add_subplot(3, 3, 8, aspect='equal')
            plt.imshow(im)
            ax.invert_yaxis()
            for i in xrange(rois.shape[0]):
                cls = int(rois[i, 1])
                if cls > 0:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses_new[i, :4])
                    RT[:, 3] = poses_new[i, 4:7]
                    print cls
                    print RT
                    print '\n'
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

            ax.set_title('projection refined by ICP')
            ax.invert_yaxis()
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])

    plt.show()



def vis_segmentations_vertmaps_3d(im, im_depth, im_labels, im_labels_gt, colors, vertmap, vertmap_target,
  labels, labels_gt, rois_rgb, poses_rgb, rois, poses, poses_icp, intrinsic_matrix, vertmap_gt, poses_gt, cls_indexes, num_classes):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 4, 1)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show gt class labels
    # ax = fig.add_subplot(3, 4, 5)
    # plt.imshow(im_labels_gt)
    # ax.set_title('gt class labels')

    # show gt vertex map
    ax = fig.add_subplot(3, 4, 6)
    plt.imshow(vertmap_target[:,:,0])
    ax.set_title('gt vertmap x')

    ax = fig.add_subplot(3, 4, 7)
    plt.imshow(vertmap_target[:,:,1])
    ax.set_title('gt vertmap y')
    
    ax = fig.add_subplot(3, 4, 8)
    plt.imshow(vertmap_target[:,:,2])
    ax.set_title('gt vertmap z')

    # show class label
    ax = fig.add_subplot(3, 4, 9)
    plt.imshow(im_labels)
    ax.set_title('class labels')

    if cfg.TEST.POSE_REG:
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
        
    # show vertex map
    ax = fig.add_subplot(3, 4, 10)
    plt.imshow(vertmap[:,:,0])
    ax.set_title('vertmap x')

    ax = fig.add_subplot(3, 4, 11)
    plt.imshow(vertmap[:,:,1])
    ax.set_title('vertmap y')
    
    ax = fig.add_subplot(3, 4, 12)
    plt.imshow(vertmap[:,:,2])
    ax.set_title('vertmap z')

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 4, 2, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(1, num_classes):
            index = np.where(labels_gt == i)
            if len(index[0]) > 0:
                num = len(index[0])
                # extract 3D points
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
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[i], 255.0), alpha=0.05)

        ax.set_title('gt projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        ax = fig.add_subplot(3, 4, 3, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois_rgb.shape[0]):
            cls = int(rois_rgb[i, 1])
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
                RT[:3, :3] = quat2mat(poses_rgb[i, :4])
                RT[:, 3] = poses_rgb[i, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

        ax.set_title('projection rgb')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

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
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

        ax.set_title('projection depth')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        if cfg.TEST.POSE_REFINE:
            ax = fig.add_subplot(3, 4, 5, aspect='equal')
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
                    RT[:3, :3] = quat2mat(poses_icp[i, :4])
                    RT[:, 3] = poses_icp[i, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

            ax.set_title('projection refined by ICP')
            ax.invert_yaxis()
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])

    plt.show()


def _get_bb2D(extent, pose, intrinsic_matrix):
    bb3d = np.zeros((3, 8), dtype=np.float32)
    
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    
    bb3d[:, 0] = [xHalf, yHalf, zHalf]
    bb3d[:, 1] = [-xHalf, yHalf, zHalf]
    bb3d[:, 2] = [xHalf, -yHalf, zHalf]
    bb3d[:, 3] = [-xHalf, -yHalf, zHalf]
    bb3d[:, 4] = [xHalf, yHalf, -zHalf]
    bb3d[:, 5] = [-xHalf, yHalf, -zHalf]
    bb3d[:, 6] = [xHalf, -yHalf, -zHalf]
    bb3d[:, 7] = [-xHalf, -yHalf, -zHalf]

    x3d = np.ones((4, 8), dtype=np.float32)
    x3d[0:3, :] = bb3d
            
    # projection
    RT = np.zeros((3, 4), dtype=np.float32)
    RT[:3, :3] = quat2mat(pose[:4])
    RT[:, 3] = pose[4:7]
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    bb = np.zeros((4,), dtype=np.float32)
    bb[0] = np.min(x2d[0, :])
    bb[1] = np.min(x2d[1, :])
    bb[2] = np.max(x2d[0, :])
    bb[3] = np.max(x2d[1, :])
    
    return bb


###################
# test single frame
###################
def test_net_single_frame(sess, net, imdb, weights_filename, model_filename):

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
    if cfg.TEST.SYNTHETIC:
        num_images = cfg.TRAIN.SYNNUM
    else:
        num_images = len(imdb.image_index)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)
    voxelizer.setup(-3, -3, -3, 3, 3, 4)

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
        # perm = xrange(num_images)
    else:
        perm = xrange(num_images)

    if cfg.TEST.SYNTHETIC:
        # perm = np.random.permutation(np.arange(cfg.TRAIN.SYNNUM))
        perm = xrange(cfg.TRAIN.SYNNUM)

        cache_file = cfg.BACKGROUND
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                backgrounds = cPickle.load(fid)
            print 'backgrounds loaded from {}'.format(cache_file)

    if (cfg.TEST.VERTEX_REG_2D and cfg.TEST.POSE_REFINE) or (cfg.TEST.VERTEX_REG_3D and cfg.TEST.POSE_REG):
        import libsynthesizer
        synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
        synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    for i in perm:

        if cfg.TEST.SYNTHETIC:
            # rgba
            filename = cfg.TRAIN.SYNROOT + '{:06d}-color.png'.format(i)
            rgba = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            if rgba.shape[2] == 4:
                # sample a background image
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
            else:
                im = rgba

            # depth
            filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(i)
            im_depth = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # label
            filename = cfg.TRAIN.SYNROOT + '{:06d}-label.png'.format(i)
            labels_gt = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # meta data
            filename = cfg.TRAIN.SYNROOT + '{:06d}-meta.mat'.format(i)
            meta_data = scipy.io.loadmat(filename)
        else:
            # parse image name
            image_index = imdb.image_index[i]

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
            if os.path.exists(imdb.depth_path_at(i)):
                im_depth = pad_im(cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED), 16)
            else:
                im_depth = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.float32)

            # read label image
            labels_gt = pad_im(cv2.imread(imdb.label_path_at(i), cv2.IMREAD_UNCHANGED), 16)

            # load meta data
            meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()

        # process annotation if training for two classes
        if imdb.num_classes == 2 and imdb._cls_index > 0:
            I = np.where(labels_gt == imdb._cls_index)
            labels_gt[:, :] = 0
            labels_gt[I[0], I[1]] = 1
            ind = np.where(meta_data['cls_indexes'] == imdb._cls_index)[0]
            meta_data['cls_indexes'] = np.ones((len(ind),), dtype=np.float32)
            if len(meta_data['poses'].shape) == 2:
                meta_data['poses'] = np.reshape(meta_data['poses'], (3, 4, 1))
            meta_data['poses'] = meta_data['poses'][:,:,ind]
            meta_data['center'] = meta_data['center'][ind,:]

        if len(labels_gt.shape) == 2:
            im_label_gt = imdb.labels_to_image(im, labels_gt)
        else:
            im_label_gt = np.copy(labels_gt[:,:,:3])
            im_label_gt[:,:,0] = labels_gt[:,:,2]
            im_label_gt[:,:,2] = labels_gt[:,:,0]

        _t['im_segment'].tic()

        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes)

        labels = unpad_im(labels, 16)
        im_scale = cfg.TEST.SCALES_BASE[0]
        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        poses_new = []
        poses_icp = []
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                # pose refinement
                fx = meta_data['intrinsic_matrix'][0, 0] * im_scale
                fy = meta_data['intrinsic_matrix'][1, 1] * im_scale
                px = meta_data['intrinsic_matrix'][0, 2] * im_scale
                py = meta_data['intrinsic_matrix'][1, 2] * im_scale
                factor = meta_data['factor_depth']
                znear = 0.25
                zfar = 6.0
                poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
                poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
                error_threshold = 0.01
                if cfg.TEST.POSE_REFINE:
                    labels_icp = labels.copy();
                    rois_icp = rois
                    if imdb.num_classes == 2:
                        I = np.where(labels_icp > 0)
                        labels_icp[I[0], I[1]] = imdb._cls_index
                        rois_icp = rois.copy()
                        rois_icp[:, 1] = imdb._cls_index
                    im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                    parameters = np.zeros((7, ), dtype=np.float32)
                    parameters[0] = fx
                    parameters[1] = fy
                    parameters[2] = px
                    parameters[3] = py
                    parameters[4] = znear
                    parameters[5] = zfar
                    parameters[6] = factor

                    height = labels_icp.shape[0]
                    width = labels_icp.shape[1]
                    num_roi = rois_icp.shape[0]
                    channel_roi = rois_icp.shape[1]
                    synthesizer.icp_python(labels_icp, im_depth, parameters, height, width, num_roi, channel_roi, \
                                           rois_icp, poses, poses_new, poses_icp, error_threshold)
                
        elif cfg.TEST.VERTEX_REG_3D:
            fx = meta_data['intrinsic_matrix'][0, 0] * im_scale
            fy = meta_data['intrinsic_matrix'][1, 1] * im_scale
            px = meta_data['intrinsic_matrix'][0, 2] * im_scale
            py = meta_data['intrinsic_matrix'][1, 2] * im_scale
            factor = meta_data['factor_depth']
            znear = 0.25
            zfar = 6.0

            # pose estimation with color only
            poses_tmp = np.zeros((3, 4, imdb.num_classes), dtype=np.float32)            
            SYN.estimate_poses_2d(labels, vertex_pred, imdb._extents, poses_tmp, imdb.num_classes, fx, fy, px, py)
            num = 0
            for j in xrange(imdb.num_classes):
                if poses_tmp[2, 3, j] > 0:
                    num += 1
            rois_rgb = np.zeros((num, 6), dtype=np.float32)
            poses_rgb = np.zeros((num, 7), dtype=np.float32)
            count = 0
            for j in xrange(imdb.num_classes):
                if poses_tmp[2, 3, j] > 0:
                    rois_rgb[count, 1] = j
                    poses_rgb[count, :4] = mat2quat(poses_tmp[:3, :3, j])
                    poses_rgb[count, 4:] = poses_tmp[:, 3, j]
                    rois_rgb[count, 2:] = _get_bb2D(imdb._extents[j, :], poses_rgb[count, :], meta_data['intrinsic_matrix']) * im_scale
                    count += 1
            print rois_rgb
            print poses_rgb

            # pose estimation with depth
            poses_tmp = np.zeros((3, 4, imdb.num_classes), dtype=np.float32)
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            SYN.estimate_poses_3d(labels, im_depth, vertex_pred, imdb._extents, poses_tmp, imdb.num_classes, fx, fy, px, py, factor)
            num = 0
            for j in xrange(imdb.num_classes):
                if poses_tmp[2, 3, j] > 0:
                    num += 1
            rois = np.zeros((num, 6), dtype=np.float32)
            poses = np.zeros((num, 7), dtype=np.float32)
            count = 0
            for j in xrange(imdb.num_classes):
                if poses_tmp[2, 3, j] > 0:
                    rois[count, 1] = j
                    poses[count, :4] = mat2quat(poses_tmp[:3, :3, j])
                    poses[count, 4:] = poses_tmp[:, 3, j]
                    rois[count, 2:] = _get_bb2D(imdb._extents[j, :], poses[count, :], meta_data['intrinsic_matrix']) * im_scale
                    count += 1
            print rois
            print poses

            # pose refinement
            poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
            poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
            error_threshold = 0.01
            if cfg.TEST.POSE_REFINE:
                labels_icp = labels.copy();
                rois_icp = rois
                if imdb.num_classes == 2:
                    I = np.where(labels_icp > 0)
                    labels_icp[I[0], I[1]] = imdb._cls_index
                    rois_icp = rois.copy()
                    rois_icp[:, 1] = imdb._cls_index
                im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                SYN.refine_poses(labels_icp, im_depth, rois_icp, poses, poses_new, poses_icp, fx, fy, px, py, znear, zfar, factor, error_threshold)

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)
        if cfg.TEST.VERTEX_REG_2D:
            seg = {'labels': labels_new, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}
        else:
            seg = {'labels': labels_new, 'rois_rgb': rois_rgb, 'poses_rgb': poses_rgb, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}
        segmentations[i] = seg
        _t['misc'].toc()

        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i, num_images, _t['im_segment'].diff, _t['misc'].diff)

        imdb.evaluate_result(i, seg, labels_gt, meta_data, output_dir)
        if cfg.TEST.VISUALIZE:
            if cfg.TEST.VERTEX_REG_2D:
                poses_gt = meta_data['poses']
                if len(poses_gt.shape) == 2:
                    poses_gt = np.reshape(poses_gt, (3, 4, 1))
                vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
                if 'vertmap' in meta_data:
                    vertmap_gt = meta_data['vertmap'].copy()
                else:
                    vertmap_gt = np.zeros((1,), dtype=np.float32)
                centers_map_gt = _vote_centers(labels_gt, meta_data['cls_indexes'].flatten(), meta_data['center'], poses_gt, imdb.num_classes, imdb._extents)
                vis_segmentations_vertmaps(im, im_depth, im_label, im_label_gt, imdb._class_colors, \
                    centers_map_gt, vertmap, labels, labels_gt, rois, poses, poses_icp, meta_data['intrinsic_matrix'], \
                    vertmap_gt, poses_gt, meta_data['cls_indexes'].flatten(), imdb.num_classes, imdb._points_all)
            elif cfg.TEST.VERTEX_REG_3D:
                poses_gt = meta_data['poses']
                if len(poses_gt.shape) == 2:
                    poses_gt = np.reshape(poses_gt, (3, 4, 1))
                vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
                vertmap_gt = meta_data['vertmap'].copy()
                vertmap_target = _generate_vertex_targets(labels_gt, imdb.num_classes, vertmap_gt, imdb._extents)
                vis_segmentations_vertmaps_3d(im, im_depth, im_label, im_label_gt, imdb._class_colors, \
                    vertmap, vertmap_target, labels, labels_gt, rois_rgb, poses_rgb, rois, poses, poses_icp, meta_data['intrinsic_matrix'], \
                    meta_data['vertmap'], poses_gt, meta_data['cls_indexes'].flatten(), imdb.num_classes)
            else:
                vis_segmentations(im, im_depth, im_label, im_label_gt, imdb._class_colors)

    '''
    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
    '''

########################
# test detection network
########################
def test_net_detection(sess, net, imdb, weights_filename):

    output_dir = get_output_dir(imdb, weights_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    det_file = os.path.join(output_dir, 'detections.pkl')
    print imdb.name
    if os.path.exists(det_file):
        with open(det_file, 'rb') as fid:
            detections = cPickle.load(fid)
        imdb.evaluate_detections(detections, output_dir)
        return

    """Test a FCN on an image database."""
    if cfg.TEST.SYNTHETIC:
        num_images = cfg.TRAIN.SYNNUM
    else:
        num_images = len(imdb.image_index)
    detections = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
        # perm = xrange(876, num_images)
    else:
        perm = xrange(num_images)

    if cfg.TEST.SYNTHETIC:
        perm = np.random.permutation(np.arange(cfg.TRAIN.SYNNUM))

        cache_file = cfg.BACKGROUND
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                backgrounds = cPickle.load(fid)
            print 'backgrounds loaded from {}'.format(cache_file)

    for i in perm:

        if cfg.TEST.SYNTHETIC:
            # rgba
            filename = cfg.TRAIN.SYNROOT + '{:06d}-color.png'.format(i)
            rgba = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # sample a background image
            ind = np.random.randint(len(backgrounds), size=1)[0]
            filename_color = backgrounds[ind]
            background_color = cv2.imread(filename_color, cv2.IMREAD_UNCHANGED)
            try:
                background_color = cv2.resize(background_color, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)
            except:
                background_color = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
                print 'bad background image'

            if len(background_color.shape) != 3:
                background_color = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
                print 'bad background image'

            # add background
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            print im.shape, background_color.shape
            im[I[0], I[1], :] = background_color[I[0], I[1], :]

            # depth
            filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(i)
            im_depth = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # meta data
            filename = cfg.TRAIN.SYNROOT + '{:06d}-meta.mat'.format(i)
            meta_data = scipy.io.loadmat(filename)
        else:
            # parse image name
            image_index = imdb.image_index[i]

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
            print imdb.metadata_path_at(i)
            meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()

        # process annotation if training for two classes
        if imdb.num_classes == 2:
            ind = np.where(meta_data['cls_indexes'] == imdb._cls_index)[0]
            if len(ind) > 0:
                meta_data['cls_indexes'] = np.ones((1,), dtype=np.float32)
                if len(meta_data['poses'].shape) == 3:
                    meta_data['poses'] = np.reshape(meta_data['poses'][:,:,ind], (3, 4, 1))
                else:
                    meta_data['poses'] = np.reshape(meta_data['poses'], (3, 4, 1))
                meta_data['center'] = meta_data['center'][ind,:]
            else:
                meta_data['cls_indexes'] = np.ones((0,), dtype=np.float32)
                meta_data['poses'] = np.zeros((3, 4, 0), dtype=np.float32)
                meta_data['center'] = np.zeros((0, 2), dtype=np.float32)

        _t['im_detect'].tic()
        boxes, scores, rois, rpn_scores, poses = im_detect_single_frame(sess, net, im, im_depth, meta_data, imdb._points_all, imdb._symmetry, imdb.num_classes)
        _t['im_detect'].toc()

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        all_dets = np.zeros((0, 6), dtype=np.float32)
        all_poses = np.zeros((0, 7), dtype=np.float32)
        thresh = 0.02
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_poses = poses[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            # only take the first one
            if len(keep) > 1:
                keep = [keep[0]]
            cls_dets = cls_dets[keep, :]
            cls_poses = cls_poses[keep, :]

            num = cls_dets.shape[0]
            if num > 0:
                cls = j * np.ones((num, 1), dtype=np.float32)
                cls_dets = np.hstack((cls, cls_dets))
                all_dets = np.vstack((all_dets, cls_dets))

                translation = np.zeros((num, 3), dtype=np.float32)
                cls_poses = np.hstack((cls_poses, translation))
                all_poses = np.vstack((all_poses, cls_poses))

        all_poses = compute_translations(all_dets, all_poses, imdb._points_all, meta_data['intrinsic_matrix'])

        det = {'rois': all_dets, 'poses': all_poses}
        detections[i] = det
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i, num_images, _t['im_detect'].diff, _t['misc'].diff)

        imdb.evaluate_result_detection(i, det, meta_data, output_dir)

        if cfg.TEST.VISUALIZE:
            vis_detections(im, im_depth, all_dets, all_poses, meta_data['cls_indexes'], meta_data['intrinsic_matrix'], \
                           meta_data['poses'], imdb._points_all, imdb._class_colors, imdb.num_classes)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(detections, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_detections(detections, output_dir)

# dets (cls, x1, y1, x2, y2, score)
def compute_translations(dets, poses, points, intrinsic_matrix):

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    num = dets.shape[0]
    # for each object
    for i in xrange(num):
        cls = int(dets[i, 0])
        # object center
        x = (dets[i, 1] + dets[i, 3]) / 2
        y = (dets[i, 2] + dets[i, 4]) / 2
        width = dets[i, 3] - dets[i, 1]
        height = dets[i, 4] - dets[i, 2]
        # backprojection
        rx = (x - cx) / fx;
        ry = (y - cy) / fy;
        x0 = 0.5
        d = minimize(distance_objective, x0, method='nelder-mead', args=(rx, ry, poses[i,:4], cls, points, intrinsic_matrix, width, height))
        poses[i, 4] = rx * d.x
        poses[i, 5] = ry * d.x
        poses[i, 6] = d.x

    return poses


def distance_objective(x, rx, ry, quaternion, cls, points, intrinsic_matrix, width, height):

    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
    x3d[0, :] = points[cls,:,0]
    x3d[1, :] = points[cls,:,1]
    x3d[2, :] = points[cls,:,2]

    # projection
    RT = np.zeros((3, 4), dtype=np.float32)
    RT[:3, :3] = quat2mat(quaternion)
    RT[0, 3] = rx * x
    RT[1, 3] = ry * x
    RT[2, 3] = x
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    # bounding box
    x1 = np.min(x2d[0, :])
    x2 = np.max(x2d[0, :])
    y1 = np.min(x2d[1, :])
    y2 = np.max(x2d[1, :])
    w = x2 - x1
    h = y2 - y1

    return (w - width) * (w - width) + (h - height) * (h - height)


def im_detect_single_frame(sess, net, im, im_depth, meta_data, points, symmetry, num_classes):
    """detect image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)
    im_scale = im_scale_factors[0]
    im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale], dtype=np.float32)

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

    # use a fake gt boxes of zeros
    gt_boxes = np.zeros((1, 5), dtype=np.float32)
    pose_blob = np.zeros((1, 13), dtype=np.float32)

    if cfg.INPUT == 'RGBD':
        feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.im_info: im_info, net.gt_boxes: gt_boxes, \
                     net.poses: pose_blob, net.points: points, net.symmetry: symmetry, net.keep_prob: 1.0}
    else:
        feed_dict = {net.data: data_blob, net.im_info: im_info, net.gt_boxes: gt_boxes, \
                     net.poses: pose_blob, net.points: points, net.symmetry: symmetry, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    
    scores, bbox_pred, rois, rpn_scores, poses = \
        sess.run([net.get_output('cls_prob'), net.get_output('bbox_pred'), net.get_output('rois'), \
                  net.get_output('rpn_scores'), net.get_output('poses_tanh')])

    stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes))
    means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes))
    bbox_pred *= stds
    bbox_pred += means

    boxes = rois[:, 1:5] / im_scale
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return pred_boxes, scores, rois, rpn_scores, poses


def vis_detections(im, im_depth, rois, poses, cls_indexes, intrinsic_matrix, poses_gt, points, colors, num_classes):
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
    plt.imshow(im)
    ax.set_title('region proposals')

    # show centers
    for i in xrange(rois.shape[0]):
        cls = int(rois[i, 0])
        x1 = rois[i, 1]
        y1 = rois[i, 2]
        w = rois[i, 3] - rois[i, 1]
        h = rois[i, 4] - rois[i, 2]
        # show boxes
        plt.gca().add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=np.array(colors[cls]) / 255.0, linewidth=3))

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(2, 2, 3, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(cls_indexes):
            cls = int(cls_indexes[i])
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = poses_gt[:, :, i]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
            plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

        ax.set_title('gt projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])


        ax = fig.add_subplot(2, 2, 4, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 0])
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
            plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

        ax.set_title('pose')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

    plt.show()


###################
# test image sets
###################
def test_net_images(sess, net, imdb, weights_filename, rgb_filenames, depth_filenames, meta_data):

    output_dir = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'output', 'images', weights_filename))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_images = len(rgb_filenames)
    segmentations = [[] for _ in xrange(num_images)]

    # timers
    _t = {'im_segment' : Timer(), 'misc' : Timer()}

    # voxelizer
    voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, imdb.num_classes)
    voxelizer.setup(-3, -3, -3, 3, 3, 4)

    # construct colors
    colors = np.zeros((3 * imdb.num_classes), dtype=np.uint8)
    for i in range(imdb.num_classes):
        colors[i * 3 + 0] = imdb._class_colors[i][0]
        colors[i * 3 + 1] = imdb._class_colors[i][1]
        colors[i * 3 + 2] = imdb._class_colors[i][2]

    if cfg.TEST.VISUALIZE:
        perm = np.random.permutation(np.arange(num_images))
        # perm = xrange(num_images)
    else:
        perm = xrange(num_images)

    if (cfg.TEST.VERTEX_REG_2D and cfg.TEST.POSE_REFINE) or (cfg.TEST.VERTEX_REG_3D and cfg.TEST.POSE_REG):
        import libsynthesizer
        synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
        synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    for i in perm:

        # read color image
        rgba = pad_im(cv2.imread(rgb_filenames[i], cv2.IMREAD_UNCHANGED), 16)
        print rgb_filenames[i]
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # read depth image
        if os.path.isfile(depth_filenames[i]):
            im_depth = pad_im(cv2.imread(depth_filenames[i], cv2.IMREAD_UNCHANGED), 16)
        else:
            im_depth = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint16)

        _t['im_segment'].tic()

        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, net, im, im_depth, meta_data, voxelizer, imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes)

        labels = unpad_im(labels, 16)
        im_scale = cfg.TEST.SCALES_BASE[0]
        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        poses_new = []
        poses_icp = []
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                # pose refinement
                fx = meta_data['intrinsic_matrix'][0, 0] * im_scale
                fy = meta_data['intrinsic_matrix'][1, 1] * im_scale
                px = meta_data['intrinsic_matrix'][0, 2] * im_scale
                py = meta_data['intrinsic_matrix'][1, 2] * im_scale
                factor = meta_data['factor_depth']
                znear = 0.25
                zfar = 6.0
                poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
                poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
                error_threshold = 0.01
                if cfg.TEST.POSE_REFINE:
                    labels_icp = labels.copy();
                    rois_icp = rois
                    if imdb.num_classes == 2:
                        I = np.where(labels_icp > 0)
                        labels_icp[I[0], I[1]] = imdb._cls_index
                        rois_icp = rois.copy()
                        rois_icp[:, 1] = imdb._cls_index
                    im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                    parameters = np.zeros((7, ), dtype=np.float32)
                    parameters[0] = fx
                    parameters[1] = fy
                    parameters[2] = px
                    parameters[3] = py
                    parameters[4] = znear
                    parameters[5] = zfar
                    parameters[6] = factor

                    height = labels_icp.shape[0]
                    width = labels_icp.shape[1]
                    num_roi = rois_icp.shape[0]
                    channel_roi = rois_icp.shape[1]
                    synthesizer.icp_python(labels_icp, im_depth, parameters, height, width, num_roi, channel_roi, \
                                           rois_icp, poses, poses_new, poses_icp, error_threshold)

        _t['im_segment'].toc()

        _t['misc'].tic()
        labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)
        seg = {'labels': labels_new, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}
        segmentations[i] = seg
        _t['misc'].toc()

        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i, num_images, _t['im_segment'].diff, _t['misc'].diff)

        imdb.save_result(i, seg, output_dir)

        if cfg.TEST.VISUALIZE:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            vis_segmentations_vertmaps_detection(im, im_depth, im_label, imdb._class_colors, vertmap, 
                labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'], imdb.num_classes, imdb._classes, imdb._points_all)


def _render_synthetic_image(SYN, num_classes, backgrounds, intrinsic_matrix):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """

    # meta data
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    znear = 0.25
    zfar = 6.0

    # sample a background image
    ind = np.random.randint(len(backgrounds), size=1)[0]
    filename = backgrounds[ind]
    background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    height = background.shape[0]
    width = background.shape[1]

    # render a synthetic image
    im_syn = np.zeros((height, width, 4), dtype=np.uint8)
    depth_syn = np.zeros((height, width), dtype=np.float32)
    vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
    class_indexes = -1 * np.ones((num_classes, ), dtype=np.float32)
    poses = np.zeros((num_classes, 7), dtype=np.float32)
    centers = np.zeros((num_classes, 2), dtype=np.float32)
    vertex_targets = np.zeros((height, width, 2*num_classes), dtype=np.float32)
    vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)
    SYN.render(im_syn, depth_syn, vertmap_syn, class_indexes, poses, centers, vertex_targets, vertex_weights, fx, fy, cx, cy, znear, zfar, 10.0)
    im_syn = im_syn[::-1, :, :]
    depth_syn = depth_syn[::-1, :]

    # convert depth
    im_depth_raw = 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * depth_syn - 1))
    I = np.where(depth_syn == 1)
    im_depth_raw[I[0], I[1]] = 0

    # add background
    alpha = im_syn[:,:,3]
    I = np.where(alpha == 0)
    im_syn[I[0], I[1], :3] = background[I[0], I[1], :]
    im = im_syn[:, :, :3]

    # compute labels from vertmap
    label = np.round(vertmap_syn[:, :, 0]) + 1
    label[np.isnan(label)] = 0

    entry = {'image': im_syn[:,:,:3],
             'label' : label,
             'depth' : im_depth_raw,
             'class_indexes' : class_indexes,
             'poses' : poses,
             'centers' : centers,
             'vertex_targets': vertex_targets,
             'vertex_weights': vertex_weights}

    return entry
