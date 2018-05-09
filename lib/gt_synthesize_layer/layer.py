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
import cv2
from utils.blob import pad_im
import os
import cPickle
import scipy.io

class GtSynthesizeLayer(object):
    """FCN data layer used for training."""

    def __init__(self, roidb, num_classes, extents, points, symmetry, cache_path, name, data_queue, model_file, pose_file):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._extents = extents
        self._points = points
        self._symmetry = symmetry
        self._cache_path = cache_path
        self._name = name
        self._data_queue = data_queue
        self._shuffle_roidb_inds()
        self._shuffle_syn_inds()
        self._shuffle_adapt_inds()
        self._build_background_images()
        self._build_background_depth_images()
        self._read_camera_parameters()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _shuffle_syn_inds(self):
        self._perm_syn = np.random.permutation(np.arange(cfg.TRAIN.SYNNUM))
        self._cur_syn = 0

    def _shuffle_adapt_inds(self):
        self._perm_adapt = np.random.permutation(np.arange(cfg.TRAIN.ADAPT_NUM))
        self._cur_adapt = 0

    def _get_next_minibatch_inds(self, is_syn, is_adapt):
        """Return the roidb indices for the next minibatch."""

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        if is_syn == 0 and is_adapt == 0:
            self._cur += cfg.TRAIN.IMS_PER_BATCH
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                self._shuffle_roidb_inds()

        db_inds_syn = self._perm_syn[self._cur_syn:self._cur_syn + cfg.TRAIN.IMS_PER_BATCH]
        if is_syn:
            self._cur_syn += cfg.TRAIN.IMS_PER_BATCH
            if self._cur_syn + cfg.TRAIN.IMS_PER_BATCH >= cfg.TRAIN.SYNNUM:
                self._shuffle_syn_inds()

        db_inds_adapt = self._perm_adapt[self._cur_adapt:self._cur_adapt + cfg.TRAIN.IMS_PER_BATCH]
        if is_adapt:
            self._cur_adapt += cfg.TRAIN.IMS_PER_BATCH
            if self._cur_adapt + cfg.TRAIN.IMS_PER_BATCH >= cfg.TRAIN.ADAPT_NUM:
                self._shuffle_adapt_inds()

        return db_inds, db_inds_syn, db_inds_adapt

    def _get_next_minibatch(self, iter):
        """Return the blobs to be used for the next minibatch."""

        if cfg.TRAIN.SYNTHESIZE:
            if cfg.TRAIN.SYN_RATIO == 0:
                is_syn = 1
            else:
                r = np.random.randint(cfg.TRAIN.SYN_RATIO+1, size=1)[0]
                if r == 0:
                    is_syn = 0
                else:
                    is_syn = 1
        else:
            is_syn = 0

        # domain adaptation
        if cfg.TRAIN.ADAPT:
            r = np.random.randint(cfg.TRAIN.ADAPT_RATIO+1, size=1)[0]
            if r == 0:
                is_adapt = 1
                is_syn = 0
            else:
                is_adapt = 0
        else:
            is_adapt = 0

        if iter >= cfg.TRAIN.SYMSIZE:
          is_symmetric = 1
        else:
          is_symmetric = 0

        db_inds, db_inds_syn, db_inds_adapt = self._get_next_minibatch_inds(is_syn, is_adapt)
        minibatch_db = [self._roidb[i] for i in db_inds]
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
            backgrounds = self._backgrounds_depth
        else:
            backgrounds = self._backgrounds
        return get_minibatch(minibatch_db, self._extents, self._points, self._symmetry, self._num_classes, backgrounds, self._intrinsic_matrix, self._data_queue, db_inds_syn, is_syn, db_inds_adapt, is_adapt, is_symmetric)
            
    def forward(self, iter):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch(iter)

        return blobs

    def _read_camera_parameters(self):
        meta_data = scipy.io.loadmat(self._roidb[0]['meta_data'])
        self._intrinsic_matrix = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)

    def _build_background_images(self):

        cache_file = os.path.join(self._cache_path, 'backgrounds.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._backgrounds = cPickle.load(fid)
            print '{} backgrounds loaded from {}'.format(self._name, cache_file)
            return

        backgrounds = []
        # SUN 2012
        root = os.path.join(self._cache_path, '../SUN2012/data/Images')
        subdirs = os.listdir(root)

        for i in xrange(len(subdirs)):
            subdir = subdirs[i]
            names = os.listdir(os.path.join(root, subdir))

            for j in xrange(len(names)):
                name = names[j]
                if os.path.isdir(os.path.join(root, subdir, name)):
                    files = os.listdir(os.path.join(root, subdir, name))
                    for k in range(len(files)):
                        if os.path.isdir(os.path.join(root, subdir, name, files[k])):
                            filenames = os.listdir(os.path.join(root, subdir, name, files[k]))
                            for l in range(len(filenames)):
                               filename = os.path.join(root, subdir, name, files[k], filenames[l])
                               backgrounds.append(filename)
                        else:
                            filename = os.path.join(root, subdir, name, files[k])
                            backgrounds.append(filename)
                else:
                    filename = os.path.join(root, subdir, name)
                    backgrounds.append(filename)

        # ObjectNet3D
        objectnet3d = os.path.join(self._cache_path, '../ObjectNet3D/data')
        files = os.listdir(objectnet3d)
        for i in range(len(files)):
            filename = os.path.join(objectnet3d, files[i])
            backgrounds.append(filename)

        for i in xrange(len(backgrounds)):
            if not os.path.isfile(backgrounds[i]):
                print 'file not exist {}'.format(backgrounds[i])

        self._backgrounds = backgrounds
        print "build background images finished"

        with open(cache_file, 'wb') as fid:
            cPickle.dump(backgrounds, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote backgrounds to {}'.format(cache_file)

    def _write_background_images(self):

        cache_file = os.path.join(self._cache_path, self._name + '_backgrounds.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._backgrounds = cPickle.load(fid)

            if self._name != 'lov_train':
                cache_file_lov = os.path.join(self._cache_path, 'lov_train_backgrounds.pkl')
                if os.path.exists(cache_file_lov):
                    with open(cache_file_lov, 'rb') as fid:
                        backgrounds_lov = cPickle.load(fid)
                        self._backgrounds = self._backgrounds + backgrounds_lov

            print '{} backgrounds loaded from {}, {} images'.format(self._name, cache_file, len(self._backgrounds))
            return

        print "building background images"

        outdir = os.path.join(self._cache_path, self._name + '_backgrounds')
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        num = 1000
        perm = np.random.permutation(np.arange(len(self._roidb)))
        perm = perm[:num]
        print len(perm)

        backgrounds = [None]*num
        kernel = np.ones((50, 50), np.uint8)
        for i in xrange(num):
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
            background = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)

            # write the image
            filename = os.path.join(self._cache_path, self._name + '_backgrounds', '%04d.jpg' % (i))
            cv2.imwrite(filename, background)
            backgrounds[i] = filename

        self._backgrounds = backgrounds
        print "build background images finished"

        with open(cache_file, 'wb') as fid:
            cPickle.dump(backgrounds, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote backgrounds to {}'.format(cache_file)


    def _build_background_depth_images(self):

        cache_file = os.path.join(self._cache_path, 'backgrounds_depth.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._backgrounds_depth = cPickle.load(fid)
            print '{} backgrounds depth loaded from {}'.format(self._name, cache_file)
            return

        backgrounds = []
        # RGBD scenes
        root = os.path.join(self._cache_path, '../RGBDScene')

        # load image index
        image_set_file = os.path.join(root, 'train.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index_train = [x.rstrip('\n') for x in f.readlines()]

        image_set_file = os.path.join(root, 'val.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index_val = [x.rstrip('\n') for x in f.readlines()]

        image_index = image_index_train + image_index_val
        print len(image_index)

        for i in range(len(image_index)):
            filename = os.path.join(root, 'data', image_index[i] + '-depth.png')
            backgrounds.append(filename)

        for i in xrange(len(backgrounds)):
            if not os.path.isfile(backgrounds[i]):
                print 'file not exist {}'.format(backgrounds[i])

        self._backgrounds_depth = backgrounds
        print "build background depth images finished"

        with open(cache_file, 'wb') as fid:
            cPickle.dump(backgrounds, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote backgrounds to {}'.format(cache_file)
