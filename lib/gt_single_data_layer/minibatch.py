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
    label_blob, depth_blob, meta_data_blob = _get_label_blob(roidb, voxelizer)

    # For debug visualizations
    # _vis_minibatch(im_blob, im_depth_blob, label_blob)

    blobs = {'data_rgb_image': im_blob,
             'data_depth_image': im_depth_blob,
             'data_label': label_blob,
             'data_depth': depth_blob,
             'data_meta_data': meta_data_blob}

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

def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)

    # label image is in BRG order
    index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
    for i in xrange(len(class_colors)):
        color = class_colors[i]
        ind = 255 * (color[0] + 256*color[1] + 256*256*color[2])
        I = np.where(index == ind)
        label_index[I[0], I[1], i] = class_weights[i]
    
    return label_index


def _get_label_blob(roidb, voxelizer):
    """ build the label blob """

    num_images = len(roidb)
    processed_label = []
    processed_depth = []
    processed_meta_data = []

    for i in xrange(num_images):
        # load meta data
        meta_data = scipy.io.loadmat(roidb[i]['meta_data'])

        # read label image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_label.append(im_cls)

        # depth
        im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED)
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]
        depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']
        processed_depth.append(depth)

        # voxelization
        points = voxelizer.backproject(im_depth, meta_data)
        voxelizer.voxelized = False
        voxelizer.voxelize(points)

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
        processed_meta_data.append(mdata)

    # construct the blobs
    height = processed_depth[0].shape[0]
    width = processed_depth[0].shape[1]
    num_classes = voxelizer.num_classes
    label_blob = np.zeros((num_images, height, width, num_classes), dtype=np.float32)
    depth_blob = np.zeros((num_images, height, width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 33), dtype=np.float32)
    for i in xrange(num_images):
        label_blob[i,:,:,:] = processed_label[i]
        depth_blob[i,:,:,0] = processed_depth[i]
        meta_data_blob[i,0,0,:] = processed_meta_data[i]

    return label_blob, depth_blob, meta_data_blob,


def _vis_minibatch(im_blob, im_depth_blob, label_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

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
        height = label.shape[0]
        width = label.shape[1]
        num_classes = label.shape[2]
        l = np.zeros((height, width), dtype=np.int32)
        for k in xrange(num_classes):
            index = np.where(label[:,:,k] > 0)
            l[index] = k
        fig.add_subplot(133)
        plt.imshow(l)

        plt.show()
