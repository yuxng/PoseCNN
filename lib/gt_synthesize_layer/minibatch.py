# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import sys
import numpy as np
import numpy.random as npr
import cv2
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, chromatic_transform
from utils.se3 import *
import scipy.io
# from normals import gpu_normals
from transforms3d.quaternions import mat2quat

def get_minibatch(roidb, extents, num_classes, backgrounds, intrinsic_matrix, db_inds_syn, is_syn):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    im_blob, im_depth_blob, im_normal_blob, im_scales = _get_image_blob(roidb, random_scale_ind, num_classes, backgrounds, intrinsic_matrix, db_inds_syn, is_syn)

    # build the label blob
    depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, \
        = _get_label_blob(roidb, intrinsic_matrix, num_classes, db_inds_syn, im_scales, extents, is_syn)

    # For debug visualizations
    if cfg.TRAIN.VISUALIZE:
        _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, vertex_target_blob)

    blobs = {'data_image_color': im_blob,
             'data_image_depth': im_depth_blob,
             'data_image_normal': im_normal_blob,
             'data_label': label_blob,
             'data_depth': depth_blob,
             'data_meta_data': meta_data_blob,
             'data_vertex_targets': vertex_target_blob,
             'data_vertex_weights': vertex_weight_blob,
             'data_pose': pose_blob,
             'data_extents': extents}

    return blobs

def _get_image_blob(roidb, scale_ind, num_classes, backgrounds, intrinsic_matrix, db_inds_syn, is_syn):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ims_depth = []
    processed_ims_normal = []
    im_scales = []
    roidb_syn = []

    for i in xrange(num_images):

        if is_syn:
            # depth raw
            filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(db_inds_syn[i])
            im_depth_raw = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # rgba
            filename = cfg.TRAIN.SYNROOT + '{:06d}-color.png'.format(db_inds_syn[i])
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
            im[I[0], I[1], :] = background_color[I[0], I[1], :3]

            if cfg.TRAIN.CHROMATIC:
                filename = cfg.TRAIN.SYNROOT + '{:06d}-label.png'.format(db_inds_syn[i])
                label = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)
        else:
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

            if cfg.TRAIN.CHROMATIC:
                label = pad_im(cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED), 16)

        # chromatic transform
        if cfg.TRAIN.CHROMATIC:
            im = chromatic_transform(im, label)

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scales.append(im_scale)
        processed_ims.append(im)

        # depth
        im_depth = im_depth_raw.astype(np.float32, copy=True) / float(im_depth_raw.max()) * 255
        im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3))

        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]

        im_orig = im_depth.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_depth = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im_depth)

        # normals
        '''
        depth = im_depth_raw.astype(np.float32, copy=True)
        nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
        im_normal = 127.5 * nmap + 127.5
        im_normal = im_normal.astype(np.uint8)
        im_normal = im_normal[:, :, (2, 1, 0)]
        if roidb[i]['flipped']:
            im_normal = im_normal[:, ::-1, :]

        im_orig = im_normal.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_normal.append(im_normal)
        '''

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)
    # blob_normal = im_list_to_blob(processed_ims_normal, 3)
    blob_normal = []

    return blob, blob_depth, blob_normal, im_scales


def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.float32)

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

            # 051_large_clamp and 052_extra_large_clamp
            if cfg.EXP_DIR == 'lov' and i == 19:
                label_index[I[0], I[1], 20] = class_weights[i]

            if cfg.EXP_DIR == 'lov' and i == 20:
                label_index[I[0], I[1], 19] = class_weights[i]
    
    return label_index, labels


def _get_label_blob(roidb, intrinsic_matrix, num_classes, db_inds_syn, im_scales, extents, is_syn):
    """ build the label blob """

    num_images = len(roidb)
    processed_depth = []
    processed_label = []
    processed_meta_data = []
    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
        processed_vertex_targets = []
        processed_vertex_weights = []
        pose_blob = np.zeros((0, 13), dtype=np.float32)
    else:
        pose_blob = []

    for i in xrange(num_images):
        im_scale = im_scales[i]

        if is_syn:
            filename = cfg.TRAIN.SYNROOT + '{:06d}-meta.mat'.format(db_inds_syn[i])
            meta_data = scipy.io.loadmat(filename)

            filename = cfg.TRAIN.SYNROOT + '{:06d}-depth.png'.format(db_inds_syn[i])
            im_depth = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)

            # read label image
            filename = cfg.TRAIN.SYNROOT + '{:06d}-label.png'.format(db_inds_syn[i])
            im = pad_im(cv2.imread(filename, cv2.IMREAD_UNCHANGED), 16)
        else:
            meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
            im_depth = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16)

            # read label image
            im = pad_im(cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED), 16)

        height = im.shape[0]
        width = im.shape[1]
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
        if num_classes == 2:
            I = np.where(im > 0)
            im[I[0], I[1]] = 1
            for j in xrange(len(meta_data['cls_indexes'])):
                meta_data['cls_indexes'][j] = 1
        im_cls, im_labels = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_label.append(im_cls)

        # vertex regression targets and weights
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            if is_syn:
                poses = meta_data['poses']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (3, 4, 1))

                vertmap = meta_data['vertmap']
                if roidb[i]['flipped']:
                    vertmap = vertmap[:, ::-1, :]
                vertmap = cv2.resize(vertmap, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                vertex_targets, vertex_weights = _generate_vertex_targets(im, meta_data['cls_indexes'].flatten(), im_scale * meta_data['center'], poses, num_classes, vertmap, extents)
                processed_vertex_targets.append(vertex_targets)
                processed_vertex_weights.append(vertex_weights)

                num = poses.shape[2]
                qt = np.zeros((num, 13), dtype=np.float32)
                for j in xrange(num):
                    R = poses[:, :3, j]
                    T = poses[:, 3, j]

                    qt[j, 0] = i
                    qt[j, 1] = meta_data['cls_indexes'][0, j]
                    qt[j, 2:6] = 0  # fill box later
                    qt[j, 6:10] = mat2quat(R)
                    qt[j, 10:] = T
            else:
                poses = meta_data['poses']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (3, 4, 1))

                vertmap = meta_data['vertmap']
                if roidb[i]['flipped']:
                    vertmap = vertmap[:, ::-1, :]
                vertmap = cv2.resize(vertmap, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                vertex_targets, vertex_weights = _generate_vertex_targets(im, meta_data['cls_indexes'], im_scale * meta_data['center'], poses, num_classes, vertmap, extents)
                processed_vertex_targets.append(vertex_targets)
                processed_vertex_weights.append(vertex_weights)

                num = poses.shape[2]
                qt = np.zeros((num, 13), dtype=np.float32)
                for j in xrange(num):
                    R = poses[:, :3, j]
                    T = poses[:, 3, j]
                    # print R, T

                    qt[j, 0] = i
                    qt[j, 1] = meta_data['cls_indexes'][j, 0]
                    qt[j, 2:6] = 0  # fill box later, roidb[i]['boxes'][j, :]
                    qt[j, 6:10] = mat2quat(R)
                    qt[j, 10:] = T

            pose_blob = np.concatenate((pose_blob, qt), axis=0)

        # depth
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]
        depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        depth = cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_depth.append(depth)

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

    # construct the blobs
    height = processed_depth[0].shape[0]
    width = processed_depth[0].shape[1]
    depth_blob = np.zeros((num_images, height, width, 1), dtype=np.float32)
    label_blob = np.zeros((num_images, height, width, num_classes), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 48), dtype=np.float32)
    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
        vertex_target_blob = np.zeros((num_images, height, width, 3 * num_classes), dtype=np.float32)
        vertex_weight_blob = np.zeros((num_images, height, width, 3 * num_classes), dtype=np.float32)
    else:
        vertex_target_blob = []
        vertex_weight_blob = []

    for i in xrange(num_images):
        depth_blob[i,:,:,0] = processed_depth[i]
        label_blob[i,:,:,:] = processed_label[i]
        meta_data_blob[i,0,0,:] = processed_meta_data[i]
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            vertex_target_blob[i,:,:,:] = processed_vertex_targets[i]
            vertex_weight_blob[i,:,:,:] = processed_vertex_weights[i]
    
    return depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob


# compute the voting label image in 2D
def _generate_vertex_targets(im_label, cls_indexes, center, poses, num_classes, vertmap, extents):
    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3*num_classes), dtype=np.float32)
    vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)

    c = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        I = np.where(im_label == i)
        if len(x) > 0:
            if cfg.TRAIN.VERTEX_REG_2D:
                ind = np.where(cls_indexes == i)[0]    
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
                vertex_targets[y, x, 3*i+2] = z
            if cfg.TRAIN.VERTEX_REG_3D:
                vertex_targets[y, x, 3*i:3*i+3] = _scale_vertmap(vertmap, I, extents[i, :])

            vertex_weights[y, x, 3*i+0] = 10.0
            vertex_weights[y, x, 3*i+1] = 10.0
            vertex_weights[y, x, 3*i+2] = 10.0

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


def _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, vertex_target_blob):
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

        # show depth image
        #depth = depth_blob[i, :, :, 0]
        #ax = fig.add_subplot(2, 3, 2)
        #plt.imshow(abs(depth))
        #ax.set_title('depth') 

        # show normal image
        im_depth = im_depth_blob[i, :, :, :].copy()
        im_depth += cfg.PIXEL_MEANS
        im_depth = im_depth[:, :, (2, 1, 0)]
        im_depth = im_depth.astype(np.uint8)
        fig.add_subplot(2, 3, 2)
        plt.imshow(im_depth)

        # show label
        label = label_blob[i, :, :, :]
        height = label.shape[0]
        width = label.shape[1]
        num_classes = label.shape[2]
        l = np.zeros((height, width), dtype=np.int32)
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((height, width, 3), dtype=np.float32)
        for k in xrange(num_classes):
            index = np.where(label[:,:,k] > 0)
            l[index] = k
            if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D and len(index[0]) > 0 and k > 0:
                center[index[0], index[1], :] = vertex_target[index[0], index[1], 3*k:3*k+3]
        ax = fig.add_subplot(2, 3, 3)
        ax.set_title('label') 
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            plt.imshow(l)
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
            plt.imshow(center[:,:,2])
            if cfg.TRAIN.VERTEX_REG_2D:
                ax.set_title('z')
            else:
                ax.set_title('vertex z')
        else:
            plt.imshow(l)

        plt.show()
