# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import datasets
from fcn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def evaluate_proposals(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        for i in xrange(num_images):
            entry = {'image' : self.roidb[i]['image'],
                     'depth' : self.roidb[i]['depth'],
                     'label' : self.roidb[i]['label'],
                     'meta_data' : self.roidb[i]['meta_data'],
                     'class_colors' : self.roidb[i]['class_colors'],
                     'class_weights' : self.roidb[i]['class_weights'],
                     'cls_index' : self.roidb[i]['cls_index'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
        print 'finish appending flipped images'

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass

    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n**2).reshape(n, n)
