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
from utils.blob import im_list_to_blob
from utils.voxelizer import Voxelizer, set_axes_equal
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import scipy.io

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


def im_segment(sess, net, im, im_depth, meta_data, voxelizer):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth)

    # depth
    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # use fake labels
    im_cls = np.zeros_like(im_depth, dtype=np.int32)

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
    label_blob = np.zeros((1, height, width, 1), dtype=np.int32)
    meta_data_blob = np.zeros((1, 1, 1, 33), dtype=np.float32)
    depth_blob[0,:,:,0] = depth
    label_blob[0,:,:,0] = im_cls
    meta_data_blob[0,0,0,:] = mdata

    # forward pass
    feed_dict = {net.data: im_depth_blob, net.depth: depth_blob, net.label: label_blob, net.meta_data: meta_data_blob}
    output = sess.run([net.get_output('label')], feed_dict=feed_dict)

    # get outputs scores: [batch_size, grid_size, grid_size, grid_size, num_classes]
    labels = output[0]

    return labels[0,:,:,0], points


def vis_segmentations(im, im_depth, labels, points):
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

    # show the 3D points
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

    plt.show()

def test_net(sess, net, imdb, weights_filename):

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

    video_index = ''
    for i in xrange(num_images):
    # for i in perm:
        # parse image name
        image_index = imdb.image_index[i]
        pos = image_index.find('/')
        if video_index == '':
            video_index = image_index[:pos]
            voxelizer.reset()
        else:
            if video_index != image_index[:pos]:
                voxelizer.reset()
                video_index = image_index[:pos]
                print 'start video {}'.format(video_index)

        rgba = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        # load meta data
        meta_data = scipy.io.loadmat(imdb.metadata_path_at(i))

        _t['im_segment'].tic()
        labels, points = im_segment(sess, net, im, im_depth, meta_data, voxelizer)
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
