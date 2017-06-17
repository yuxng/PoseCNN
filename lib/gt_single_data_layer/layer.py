# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a FCN for single frames.
"""

from fcn.config import cfg
from gt_single_data_layer.minibatch import get_minibatch
import numpy as np
from utils.voxelizer import Voxelizer

class GtSingleDataLayer(object):
    """FCN data layer used for training."""

    def __init__(self, roidb, num_classes, extents):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._extents = extents;
        self._voxelizer = Voxelizer(cfg.TRAIN.GRID_SIZE, num_classes)
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        #print self._roidb[self._perm[14]]['meta_data']
        #print self._roidb[self._perm[15]]['meta_data']
        #print self._roidb[self._perm[16]]['meta_data']
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
        return get_minibatch(minibatch_db, self._voxelizer, self._extents)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs
