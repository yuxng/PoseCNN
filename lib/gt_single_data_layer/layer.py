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

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._voxelizer = Voxelizer(cfg.TRAIN.GRID_SIZE, num_classes)
        
        # build a dictionary of the videos
        videos = dict()
        for i in xrange(len(roidb)):
            video_id = roidb[i]['video_id']
            if video_id in videos:
                videos[video_id].extend([i])
            else:
                videos[video_id] = [i]

        video_ids = []
        for key, value in videos.iteritems() :
            video_ids.append(key)
        print video_ids

        self._videos = videos
        self._video_ids = video_ids
        self._num_videos = len(videos)
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # shuffle videos
        index_video = np.random.permutation(np.arange(self._num_videos))
        # shuffle frames
        indexes_frame = []
        for i in xrange(self._num_videos):
            indexes_frame.append(np.random.permutation(np.arange(len(self._videos[self._video_ids[i]]))))

        pv = 0
        pf = np.zeros((self._num_videos,), dtype=np.int32)
        perm = np.zeros((len(self._roidb),), dtype=np.int32)
        for i in xrange(len(self._roidb)):
            vpos = index_video[pv]
            fpos = pf[vpos]
            perm[i] = self._videos[self._video_ids[vpos]][ indexes_frame[vpos][fpos] ]

            if fpos + 1 < len(self._videos[self._video_ids[vpos]]):
                pf[vpos] += 1
            else:
                pf[vpos] = 0

            pv = pv + 1
            if pv >= self._num_videos:
                pv = 0

        # self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._perm = perm
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
        return get_minibatch(minibatch_db, self._voxelizer)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs
