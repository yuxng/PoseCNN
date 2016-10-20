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
from utils.blob import im_list_to_blob
import scipy.io

def get_minibatch(roidb, voxelizer):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    im_blob, im_depth_blob, im_scales = _get_image_blob(roidb, random_scale_ind)

    # build the label blob
    location_blob, label_blob = _get_label_blob(roidb, voxelizer)

    # For debug visualizations
    # _vis_minibatch(im_blob, im_depth_blob, label_blob, voxelizer)

    blobs = {'data_image': im_blob,
             'data_depth': im_depth_blob,
             'data_label': label_blob,
             'data_location': location_blob}

    return blobs

def _get_image_blob(roidb, scale_ind):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ims_depth = []
    im_scales = []
    for i in xrange(num_images):
        # rgba
        rgba = cv2.imread(roidb[i]['image'], cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scales.append(im_scale)
        processed_ims.append(im)

        # depth
        im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        im_depth = im_depth / im_depth.max() * 255
        im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3))
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]

        im_orig = im_depth.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_depth = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im_depth)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, blob_depth, im_scales

def _process_label_image(label_image, class_colors):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    label_index = np.zeros((height, width), dtype=np.int32)

    # label image is in BRG order
    index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
    for i in xrange(len(class_colors)):
        color = class_colors[i]
        ind = 255 * (color[0] + 256*color[1] + 256*256*color[2])
        I = np.where(index == ind)
        label_index[I] = i
    
    return label_index


def _get_label_blob(roidb, voxelizer):
    """ build the label blob """

    num_images = len(roidb)
    grid_size = voxelizer.grid_size
    filter_h = voxelizer.filter_h
    filter_w = voxelizer.filter_w
    processed_locations = np.zeros((num_images, grid_size, grid_size, grid_size, filter_h*filter_w), dtype=np.int32)
    processed_labels = np.zeros((num_images, grid_size, grid_size, grid_size), dtype=np.int32)

    for i in xrange(num_images):
        # read label image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_cls = _process_label_image(im, roidb[i]['class_colors'])

        # backproject the labels into 3D voxel space
        # depth
        im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED)

        # load meta data
        meta_data = scipy.io.loadmat(roidb[i]['meta_data'])

        # backprojection
        points = voxelizer.backproject(im_depth, meta_data)
        voxelizer.voxelized = False
        grid_indexes = voxelizer.voxelize(points)
        locations, labels = voxelizer.compute_voxel_labels(grid_indexes, im_cls, meta_data['projection_matrix'], cfg.GPU_ID)

        processed_locations[i,:,:,:,:] = locations
        processed_labels[i,:,:,:] = labels

    return processed_locations, processed_labels


def _vis_minibatch(im_blob, im_depth_blob, label_blob, voxelizer):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    from utils.voxelizer import set_axes_equal

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        fig.add_subplot(131)
        plt.imshow(im)

        # show depth image
        im_depth = im_depth_blob[i, :, :, :].copy()
        im_depth += cfg.PIXEL_MEANS
        im_depth = im_depth[:, :, (2, 1, 0)]
        im_depth = im_depth.astype(np.uint8)
        fig.add_subplot(132)
        plt.imshow(im_depth)

        # show label
        label = label_blob[i, :, :, :]
        index = np.where(label == 1)
        X = index[0] * voxelizer.step_x + voxelizer.min_x
        Y = index[1] * voxelizer.step_y + voxelizer.min_y
        Z = index[2] * voxelizer.step_z + voxelizer.min_z
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(X, Y, Z, c='r', marker='o')

        index = np.where(label > 1)
        X = index[0] * voxelizer.step_x + voxelizer.min_x
        Y = index[1] * voxelizer.step_y + voxelizer.min_y
        Z = index[2] * voxelizer.step_z + voxelizer.min_z
        ax.scatter(X, Y, Z, c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        set_axes_equal(ax)

        plt.show()
