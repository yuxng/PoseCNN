# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.
"""

from fcn.config import cfg
from gt_data_layer.minibatch import get_minibatch
import numpy as np
from utils.voxelizer import Voxelizer

class GtDataLayer(object):
    """FCN data layer used for training."""

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._voxelizer = Voxelizer(cfg.TRAIN.GRID_SIZE, num_classes)
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        num_steps = cfg.TRAIN.NUM_STEPS
        ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
        db_inds = np.zeros(num_steps * ims_per_batch, dtype=np.int32)
        interval = 1
        count = 0
        while count < ims_per_batch:
            ind = self._perm[self._cur]
            if ind + (num_steps - 1) * interval < len(self._roidb) and self._roidb[ind]['video_id'] == self._roidb[ind + (num_steps-1) * interval]['video_id']:
                db_inds[count * num_steps : (count+1) * num_steps] = range(ind, ind + num_steps * interval, interval)
                count += 1
            self._cur += 1
            if self._cur >= len(self._roidb):
                self._shuffle_roidb_inds()

        db_inds_reorder = np.zeros(num_steps * ims_per_batch, dtype=np.int32)
        count = 0
        for i in xrange(num_steps):
            for j in xrange(ims_per_batch):
                db_inds_reorder[count] = db_inds[j * num_steps + i]
                count = count + 1

        return db_inds_reorder

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._voxelizer)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs
