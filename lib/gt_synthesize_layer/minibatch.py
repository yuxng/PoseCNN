# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import sys
import os
import numpy as np
import numpy.random as npr
import cv2
import math
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, chromatic_transform, add_noise
from utils.se3 import *
import scipy.io
from normals import gpu_normals
from transforms3d.quaternions import mat2quat, quat2mat
from utils.timer import Timer

def get_minibatch(roidb, extents, points, symmetry, num_classes, backgrounds, intrinsic_matrix, \
    data_queue, db_inds_syn, is_syn, db_inds_adapt, is_adapt, is_symmetric):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    im_blob, im_depth_blob, im_normal_blob, im_scales, data_out, height, width = _get_image_blob(roidb, random_scale_ind, num_classes, backgrounds, intrinsic_matrix, data_queue, db_inds_syn, is_syn, db_inds_adapt, is_adapt)

    # build the label blob
    depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes \
        = _get_label_blob(roidb, intrinsic_matrix, data_out, num_classes, db_inds_syn, im_scales, extents, is_syn, db_inds_adapt, is_adapt, height, width)

    if not cfg.TRAIN.SEGMENTATION:
        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    else:
        im_info = []

    # For debug visualizations
    if cfg.TRAIN.VISUALIZE:
        if cfg.TRAIN.SEGMENTATION:
            _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, extents)
        else:
            _vis_minibatch_box(im_blob, gt_boxes)

    # rescale the points
    point_blob = points.copy()
    for i in xrange(1, num_classes):
        # compute the rescaling factor for the points
        weight = 2.0 / np.amax(extents[i, :])
        if weight < 10:
            weight = 10
        if symmetry[i] > 0 and is_symmetric:
            point_blob[i, :, :] = 4 * weight * point_blob[i, :, :]
        else:
            point_blob[i, :, :] = weight * point_blob[i, :, :]

    if is_symmetric:
        symmetry_blob = symmetry
    else:
        symmetry_blob = np.zeros_like(symmetry)

    blobs = {'data_image_color': im_blob,
             'data_image_depth': im_depth_blob,
             'data_image_normal': im_normal_blob,
             'data_label': label_blob,
             'data_depth': depth_blob,
             'data_meta_data': meta_data_blob,
             'data_vertex_targets': vertex_target_blob,
             'data_vertex_weights': vertex_weight_blob,
             'data_pose': pose_blob,
             'data_extents': extents,
             'data_points': point_blob,
             'data_symmetry': symmetry_blob,
             'data_gt_boxes': gt_boxes,
             'data_im_info': im_info}

    return blobs

def _get_image_blob(roidb, scale_ind, num_classes, backgrounds, intrinsic_matrix, data_queue, db_inds_syn, is_syn, db_inds_adapt, is_adapt):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ims_depth = []
    processed_ims_normal = []
    im_scales = []
    roidb_syn = []
    data_out = []

    for i in xrange(num_images):

        if is_adapt:
            if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD' or cfg.INPUT == 'NORMAL':
                # depth raw
                filename = cfg.TRAIN.ADAPT_ROOT + '{:06d}-depth.png'.format(db_inds_adapt[i])
                im_depth_raw = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # rgba
            filename = cfg.TRAIN.ADAPT_ROOT + '{:06d}-color.png'.format(db_inds_adapt[i])
            rgba = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)
            if rgba.shape[2] == 4:
                im = np.copy(rgba[:,:,:3])
                alpha = rgba[:,:,3]
                I = np.where(alpha == 0)
                im[I[0], I[1], :] = 0
            else:
                im = rgba
        else:
            if is_syn:
                if cfg.TRAIN.SYN_ONLINE:
                    data = data_queue.get()
                    data_out.append(data)
                    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD' or cfg.INPUT == 'NORMAL':
                        im_depth_raw = pad_im(data['depth'], 16)
                    rgba = pad_im(data['image'], 16)
                else:
                    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD' or cfg.INPUT == 'NORMAL':
                        # depth raw
                        filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(db_inds_syn[i])
                        im_depth_raw = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

                    # rgba
                    filename = cfg.TRAIN.SYNROOT + '{:06d}-color.png'.format(db_inds_syn[i])
                    rgba = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

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
                    im_depth_raw[I[0], I[1]] = background[I[0], I[1]] / 10
                else:
                    im[I[0], I[1], :] = background[I[0], I[1], :3]
            else:
                if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD' or cfg.INPUT == 'NORMAL':
                    # depth raw
                    im_depth_raw = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16)

                # rgba
                rgba = pad_im(cv2.imread(roidb[i]['image'], cv2.IMREAD_UNCHANGED), 16)
                if rgba.shape[2] == 4:
                    im = np.copy(rgba[:,:,:3])
                    alpha = rgba[:,:,3]
                    I = np.where(alpha == 0)
                    im[I[0], I[1], :] = 0
                else:
                    im = rgba

        # chromatic transform
        if cfg.TRAIN.CHROMATIC:
            im = chromatic_transform(im)

        if cfg.TRAIN.ADD_NOISE:
            im = add_noise(im)

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scales.append(im_scale)
        processed_ims.append(im)

        # depth
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            im_depth = im_depth_raw.astype(np.float32, copy=True) / float(im_depth_raw.max()) * 255
            im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3))

            if cfg.TRAIN.ADD_NOISE:
                im_depth = add_noise(im_depth)

            if roidb[i]['flipped']:
                im_depth = im_depth[:, ::-1]

            im_orig = im_depth.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_depth = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims_depth.append(im_depth)

        # normals
        if cfg.INPUT == 'NORMAL':
            depth = im_depth_raw.astype(np.float32, copy=True) / 1000.0
            fx = intrinsic_matrix[0, 0] * im_scale
            fy = intrinsic_matrix[1, 1] * im_scale
            cx = intrinsic_matrix[0, 2] * im_scale
            cy = intrinsic_matrix[1, 2] * im_scale
            nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
            im_normal = 127.5 * nmap + 127.5
            im_normal = im_normal.astype(np.uint8)
            im_normal = im_normal[:, :, (2, 1, 0)]
            im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)
            if roidb[i]['flipped']:
                im_normal = im_normal[:, ::-1, :]

            im_orig = im_normal.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims_normal.append(im_normal)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        blob_depth = im_list_to_blob(processed_ims_depth, 3)
    else:
        blob_depth = []

    if cfg.INPUT == 'NORMAL':
        blob_normal = im_list_to_blob(processed_ims_normal, 3)
    else:
        blob_normal = []

    height = processed_ims[0].shape[0]
    width = processed_ims[0].shape[1]

    return blob, blob_depth, blob_normal, im_scales, data_out, height, width


def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.int32)

    if len(label_image.shape) == 3:
        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I[0], I[1], i] = class_weights[i]
            labels[I[0], I[1]] = i
    else:
        for i in xrange(len(class_colors)):
            I = np.where(label_image == i)
            label_index[I[0], I[1], i] = class_weights[i]
            labels[I[0], I[1]] = i
    
    return label_index, labels


def _get_label_blob(roidb, intrinsic_matrix, data_out, num_classes, db_inds_syn, im_scales, extents, \
    is_syn, db_inds_adapt, is_adapt, blob_height, blob_width):
    """ build the label blob """

    num_images = len(roidb)
    processed_depth = []
    processed_label = []
    processed_meta_data = []
    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
        vertex_target_blob = np.zeros((num_images, blob_height, blob_width, 3 * num_classes), dtype=np.float32)
        vertex_weight_blob = np.zeros((num_images, blob_height, blob_width, 3 * num_classes), dtype=np.float32)
        pose_blob = np.zeros((0, 13), dtype=np.float32)
    else:
        vertex_target_blob = []
        vertex_weight_blob = []
        pose_blob = []

    if not cfg.TRAIN.SEGMENTATION:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

    if not cfg.TRAIN.SEGMENTATION:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_boxes = np.zeros((0, 5), dtype=np.float32)
        pose_blob = np.zeros((0, 13), dtype=np.float32)
    else:
        gt_boxes = []

    for i in xrange(num_images):
        im_scale = im_scales[i]

        if is_adapt:
            filename = cfg.TRAIN.ADAPT_ROOT + '{:06d}-depth.png'.format(db_inds_adapt[i])
            im_depth = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)
            meta_data = dict({'intrinsic_matrix': intrinsic_matrix, 'factor_depth': 1000.0})
        else:
            if is_syn:
                if cfg.TRAIN.SYN_ONLINE:
                    meta_data = data_out[i]['meta_data']
                    meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
                    im_depth = pad_im(data_out[i]['depth'], 16)
                    im = pad_im(data_out[i]['label'], 16)
                else:
                    filename = cfg.TRAIN.SYNROOT + '{:06d}-meta.mat'.format(db_inds_syn[i])
                    meta_data = scipy.io.loadmat(filename)
                    meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()

                    filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(db_inds_syn[i])
                    im_depth = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

                    # read label image
                    filename = cfg.TRAIN.SYNROOT + '{:06d}-label.png'.format(db_inds_syn[i])
                    im = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)
            else:
                meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
                meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
                if os.path.exists(roidb[i]['depth']):
                    im_depth = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16)
                else:
                    im_depth = np.zeros((blob_height, blob_width), dtype=np.float32)

                # read label image
                im = pad_im(cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED), 16)

            height = im_depth.shape[0]
            width = im_depth.shape[1]

            # mask the label image according to depth
            if cfg.INPUT == 'DEPTH':
                I = np.where(im_depth == 0)
                if len(im.shape) == 2:
                    im[I[0], I[1]] = 0
                else:
                    im[I[0], I[1], :] = 0

            if roidb[i]['flipped']:
                if len(im.shape) == 2:
                    im = im[:, ::-1]
                else:
                    im = im[:, ::-1, :]
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

            # process annotation if training for two classes
            cls_indexes_old = []
            if num_classes == 2 and roidb[i]['cls_index'] > 0:
                I = np.where(im == roidb[i]['cls_index'])
                im[:, :] = 0
                im[I[0], I[1]] = 1
                ind = np.where(meta_data['cls_indexes'] == roidb[i]['cls_index'])[0]
                cls_indexes_old = ind
                meta_data['cls_indexes'] = np.ones((len(ind),), dtype=np.float32)
                if len(meta_data['poses'].shape) == 2:
                    meta_data['poses'] = np.reshape(meta_data['poses'], (3, 4, 1))
                meta_data['poses'] = meta_data['poses'][:,:,ind]
                meta_data['center'] = meta_data['center'][ind,:]
                meta_data['box'] = meta_data['box'][ind,:]

            # im_cls, im_labels = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
            im_labels = im.copy()
            processed_label.append(im_labels.astype(np.int32))

            # bounding boxes
            if not cfg.TRAIN.SEGMENTATION:
                boxes = meta_data['box'].copy()
                if roidb[i]['flipped']:
                    oldx1 = boxes[:, 0].copy()
                    oldx2 = boxes[:, 2].copy()
                    boxes[:, 0] = width - oldx2 - 1
                    boxes[:, 2] = width - oldx1 - 1
                gt_box = np.concatenate((boxes * im_scales[0], meta_data['cls_indexes'][:, np.newaxis]), axis=1)
                gt_boxes = np.concatenate((gt_boxes, gt_box), axis=0)

                poses = meta_data['poses']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (3, 4, 1))
                if roidb[i]['flipped']:
                    poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

                num = poses.shape[2]
                qt = np.zeros((num, 13), dtype=np.float32)
                for j in xrange(num):
                    R = poses[:, :3, j]
                    T = poses[:, 3, j]

                    qt[j, 0] = i
                    qt[j, 1] = meta_data['cls_indexes'][j]
                    qt[j, 2:6] = 0  # fill box later
                    qt[j, 6:10] = mat2quat(R)
                    qt[j, 10:] = T

                pose_blob = np.concatenate((pose_blob, qt), axis=0)

            # vertex regression targets and weights
            if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
                poses = meta_data['poses']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (3, 4, 1))
                if roidb[i]['flipped']:
                    poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

                if cfg.TRAIN.VERTEX_REG_3D:
                    vertmap = meta_data['vertmap']
                    if roidb[i]['flipped']:
                        vertmap = vertmap[:, ::-1, :]
                    vertmap = cv2.resize(vertmap, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                else:
                    vertmap = []

                center = meta_data['center']
                if roidb[i]['flipped']:
                    center[:, 0] = width - center[:, 0]

                # check if mutiple same instances
                cls_indexes = meta_data['cls_indexes']
                if len(np.unique(cls_indexes)) < len(cls_indexes):
                    is_multi_instances = 1
                    # read mask image
                    mask = pad_im(cv2.imread(roidb[i]['mask'], cv2.IMREAD_UNCHANGED), 16)
                else:
                    is_multi_instances = 0
                    mask = []

                vertex_target_blob[i,:,:,:], vertex_weight_blob[i,:,:,:] = \
                    _generate_vertex_targets(im, meta_data['cls_indexes'], im_scale * center, poses, num_classes, vertmap, extents, \
                                             mask, is_multi_instances, cls_indexes_old, \
                                             vertex_target_blob[i,:,:,:], vertex_weight_blob[i,:,:,:])

                num = poses.shape[2]
                qt = np.zeros((num, 13), dtype=np.float32)
                for j in xrange(num):
                    R = poses[:, :3, j]
                    T = poses[:, 3, j]

                    qt[j, 0] = i
                    qt[j, 1] = meta_data['cls_indexes'][j]
                    qt[j, 2:6] = 0  # fill box later
                    qt[j, 6:10] = mat2quat(R)
                    qt[j, 10:] = T

                pose_blob = np.concatenate((pose_blob, qt), axis=0)

            # voxelization
            # points = voxelizer.backproject_camera(im_depth, meta_data)
            # voxelizer.voxelized = False
            # voxelizer.voxelize(points)
            # RT_world = meta_data['rotation_translation_matrix']

            # compute camera poses
            # RT_live = meta_data['rotation_translation_matrix']
            # pose_world2live = se3_mul(RT_live, se3_inverse(RT_world))
            # pose_live2world = se3_inverse(pose_world2live)

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
        # mdata[42] = voxelizer.step_x
        # mdata[43] = voxelizer.step_y
        # mdata[44] = voxelizer.step_z
        # mdata[45] = voxelizer.min_x
        # mdata[46] = voxelizer.min_y
        # mdata[47] = voxelizer.min_z
        if cfg.FLIP_X:
            mdata[0] = -1 * mdata[0]
            mdata[9] = -1 * mdata[9]
            mdata[11] = -1 * mdata[11]
        processed_meta_data.append(mdata)

        # depth
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]
        depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        depth = cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_depth.append(depth)


    # construct the blobs
    depth_blob = np.zeros((num_images, blob_height, blob_width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 48), dtype=np.float32)

    for i in xrange(num_images):
        depth_blob[i,:,:,0] = processed_depth[i]
        meta_data_blob[i,0,0,:] = processed_meta_data[i]

    if is_adapt:
        label_blob = -1 * np.ones((num_images, blob_height, blob_width), dtype=np.int32)
    else:
        label_blob = np.zeros((num_images, blob_height, blob_width), dtype=np.int32)

        for i in xrange(num_images):
            label_blob[i,:,:] = processed_label[i]

        # filter bad boxes
        if not cfg.TRAIN.SEGMENTATION:
            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
            ind = np.where((gt_widths > 0) & (gt_heights > 0))[0]
            gt_boxes = gt_boxes[ind, :]
    
    return depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes


def _flip_poses(poses, K, width):
    K1 = K.copy()
    K1[0, 0] = -1 * K1[0, 0]
    K1[0, 2] = width - K1[0, 2]

    num = poses.shape[2]
    poses_new = poses.copy()
    for i in xrange(num):
        pose = poses[:, :, i]
        poses_new[:, :, i] = np.matmul(np.linalg.inv(K), np.matmul(K1, pose))

    return poses_new


# compute the voting label image in 2D
def _generate_vertex_targets(im_label, cls_indexes, center, poses, num_classes, vertmap, extents, \
    mask, is_multi_instances, cls_indexes_old, vertex_targets, vertex_weights):

    width = im_label.shape[1]
    height = im_label.shape[0]

    if is_multi_instances:
        c = np.zeros((2, 1), dtype=np.float32)
        for i in xrange(len(cls_indexes)):
            cls = int(cls_indexes[i])
            y, x = np.where((mask == cls_indexes_old[i]+1) & (im_label == cls))
            I = np.where((mask == cls_indexes_old[i]+1) & (im_label == cls))
            if len(x) > 0:
                if cfg.TRAIN.VERTEX_REG_2D:
                    c[0] = center[i, 0]
                    c[1] = center[i, 1]
                    z = poses[2, 3, i]
                    R = np.tile(c, (1, len(x))) - np.vstack((x, y))
                    # compute the norm
                    N = np.linalg.norm(R, axis=0) + 1e-10
                    # normalization
                    R = np.divide(R, np.tile(N, (2,1)))
                    # assignment
                    vertex_targets[y, x, 3*cls+0] = R[0,:]
                    vertex_targets[y, x, 3*cls+1] = R[1,:]
                    vertex_targets[y, x, 3*cls+2] = math.log(z)
                if cfg.TRAIN.VERTEX_REG_3D:
                    vertex_targets[y, x, 3*cls:3*cls+3] = _scale_vertmap(vertmap, I, extents[cls, :])

                vertex_weights[y, x, 3*cls+0] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[y, x, 3*cls+1] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[y, x, 3*cls+2] = cfg.TRAIN.VERTEX_W_INSIDE
    else:
        c = np.zeros((2, 1), dtype=np.float32)
        for i in xrange(1, num_classes):
            y, x = np.where(im_label == i)
            I = np.where(im_label == i)
            ind = np.where(cls_indexes == i)[0]
            if len(x) > 0 and len(ind) > 0:
                if cfg.TRAIN.VERTEX_REG_2D:
                    c[0] = center[ind, 0]
                    c[1] = center[ind, 1]
                    z = poses[2, 3, ind]
                    R = np.tile(c, (1, len(x))) - np.vstack((x, y))
                    # compute the norm
                    N = np.linalg.norm(R, axis=0) + 1e-10
                    # normalization
                    R = np.divide(R, np.tile(N, (2,1)))
                    # assignment
                    vertex_targets[y, x, 3*i+0] = R[0,:]
                    vertex_targets[y, x, 3*i+1] = R[1,:]
                    vertex_targets[y, x, 3*i+2] = math.log(z)
                if cfg.TRAIN.VERTEX_REG_3D:
                    vertex_targets[y, x, 3*i:3*i+3] = _scale_vertmap(vertmap, I, extents[i, :])

                vertex_weights[y, x, 3*i+0] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[y, x, 3*i+1] = cfg.TRAIN.VERTEX_W_INSIDE
                vertex_weights[y, x, 3*i+2] = cfg.TRAIN.VERTEX_W_INSIDE

    return vertex_targets, vertex_weights


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


def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)
    
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    
    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]
    
    return bb


def _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, extents):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(2, 3, 1)
        plt.imshow(im)
        ax.set_title('color') 

        # project the 3D box to image
        metadata = meta_data_blob[i, 0, 0, :]
        intrinsic_matrix = metadata[:9].reshape((3,3))
        for j in xrange(pose_blob.shape[0]):
            if pose_blob[j, 0] != i:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose_blob[j, 6:10])
            RT[:, 3] = pose_blob[j, 10:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show depth image
        #depth = depth_blob[i, :, :, 0]
        #ax = fig.add_subplot(2, 3, 2)
        #plt.imshow(abs(depth))
        #ax.set_title('depth') 

        # show depth image
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            im_depth = im_depth_blob[i, :, :, :].copy()
            im_depth += cfg.PIXEL_MEANS
            im_depth = im_depth[:, :, (2, 1, 0)]
            im_depth = im_depth.astype(np.uint8)
            ax = fig.add_subplot(2, 3, 2)
            plt.imshow(im_depth)
            ax.set_title('depth') 

        # show label
        label = label_blob[i, :, :]
        height = label.shape[0]
        width = label.shape[1]
        num_classes = vertex_target_blob.shape[3] / 3
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((height, width, 3), dtype=np.float32)
        for k in xrange(num_classes):
            index = np.where(label == k)
            if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D and len(index[0]) > 0 and k > 0:
                center[index[0], index[1], :] = vertex_target[index[0], index[1], 3*k:3*k+3]
        ax = fig.add_subplot(2, 3, 3)
        ax.set_title('label') 
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            plt.imshow(label)
            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(center[:,:,0])
            if cfg.TRAIN.VERTEX_REG_2D:
                ax.set_title('center x') 
            else:
                ax.set_title('vertex x') 
            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(center[:,:,1])
            if cfg.TRAIN.VERTEX_REG_2D:
                ax.set_title('center y')
            else:
                ax.set_title('vertex y')
            ax = fig.add_subplot(2, 3, 6)
            plt.imshow(np.exp(center[:,:,2]))
            if cfg.TRAIN.VERTEX_REG_2D:
                ax.set_title('z')
            else:
                ax.set_title('vertex z')
        else:
            plt.imshow(l)

        plt.show()


def _vis_minibatch_box(im_blob, gt_boxes):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(im)
        ax.set_title('color') 

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(im)
        for j in xrange(gt_boxes.shape[0]):
            x1 = gt_boxes[j, 0]
            y1 = gt_boxes[j, 1]
            x2 = gt_boxes[j, 2]
            y2 = gt_boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        plt.show()
