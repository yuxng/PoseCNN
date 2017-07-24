# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a FCN for single frames.
"""

from fcn.config import cfg
from gt_synthesize_layer.minibatch import get_minibatch
import numpy as np
from synthesize import synthesizer
import cv2
from utils.blob import pad_im
import os
import cPickle
import scipy.io

class GtSynthesizeLayer(object):
    """FCN data layer used for training."""

    def __init__(self, roidb, num_classes, extents, cache_path, name, model_file, pose_file):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._extents = extents
        self._cache_path = cache_path
        self._name = name
        self._shuffle_roidb_inds()
        self._synthesizer = synthesizer.PySynthesizer(model_file, pose_file)
        self._build_background_images()
        self._read_camera_parameters()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._extents, self._synthesizer, self._num_classes, self._backgrounds, self._intrinsic_matrix)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs

    def _read_camera_parameters(self):
        meta_data = scipy.io.loadmat(self._roidb[0]['meta_data'])
        self._intrinsic_matrix = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)

    def _build_background_images(self):

        cache_file = os.path.join(self._cache_path, self._name + '_backgrounds.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._backgrounds = cPickle.load(fid)
            print '{} backgrounds loaded from {}'.format(self._name, cache_file)
            return

        print "building background images"
        num = 10
        perm = np.random.permutation(np.arange(len(self._roidb)))
        perm = perm[:num]
        print len(perm)

        im = pad_im(cv2.imread(self._roidb[0]['image'], cv2.IMREAD_UNCHANGED), 16)
        height = im.shape[0]
        width = im.shape[1]
        backgrounds = np.zeros((num, height, width, 3), dtype=np.uint8)

        kernel = np.ones((50, 50), np.uint8)
        for i in xrange(num):
            print i
            index = perm[i]
            # rgba
            rgba = pad_im(cv2.imread(self._roidb[index]['image'], cv2.IMREAD_UNCHANGED), 16)
            if rgba.shape[2] == 4:
                im = np.copy(rgba[:,:,:3])
                alpha = rgba[:,:,3]
                I = np.where(alpha == 0)
                im[I[0], I[1], :] = 0
            else:
                im = rgba

            # generate background image
            mask = pad_im(cv2.imread(self._roidb[index]['label'], cv2.IMREAD_UNCHANGED), 16)
            index = np.where(mask > 0)
            mask[index[0], index[1]] = 1
            mask = cv2.dilate(mask, kernel)
            backgrounds[i, :, :, :]  = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)

        self._backgrounds = backgrounds
        print "build background images finished"

        with open(cache_file, 'wb') as fid:
            cPickle.dump(backgrounds, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote backgrounds to {}'.format(cache_file)
