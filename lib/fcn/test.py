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
import numpy as np
import cv2
import cPickle
from utils.blob import im_list_to_blob
import os
import math
import tensorflow as tf

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


def im_segment(sess, net, im, im_depth):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_scale_factors = _get_image_blob(im, im_depth)

    # forward pass
    feed_dict = {net.data: im_depth_blob}
    output = sess.run([net.get_output('score_up')], feed_dict=feed_dict)

    # get outputs
    cls_prob = output[0]
    height = cls_prob.shape[1]
    width = cls_prob.shape[2]
    cls_label = np.argmax(cls_prob, axis = 3).reshape((height, width))

    # rescale labels
    image_height = im.shape[0]
    image_width = im.shape[1]
    labels = cv2.resize(cls_label, dsize=(image_width, image_height), interpolation=cv2.INTER_NEAREST)

    return labels


def vis_segmentations(im, im_depth, labels):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(131)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(132)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(133)
    plt.imshow(labels)
    ax.set_title('class labels')

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

    # perm = np.random.permutation(np.arange(num_images))

    for i in xrange(num_images):
    # for i in perm:
        rgba = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
        im = rgba[:,:,:3]
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 255

        im_depth = cv2.imread(imdb.depth_path_at(i), cv2.IMREAD_UNCHANGED)

        _t['im_segment'].tic()
        labels = im_segment(sess, net, im, im_depth)
        _t['im_segment'].toc()

        _t['misc'].tic()
        seg = {'labels': labels}
        segmentations[i] = seg
        _t['misc'].toc()

        # vis_segmentations(im, im_depth, labels)
        print 'im_segment: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_segment'].average_time, _t['misc'].average_time)

    seg_file = os.path.join(output_dir, 'segmentations.pkl')
    with open(seg_file, 'wb') as f:
        cPickle.dump(segmentations, f, cPickle.HIGHEST_PROTOCOL)

    # evaluation
    imdb.evaluate_segmentations(segmentations, output_dir)
