# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a FCN"""

from fcn.config import cfg
from gt_data_layer.layer import GtDataLayer
from gt_single_data_layer.layer import GtSingleDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        # For checkpoint
        self.saver = tf.train.Saver()


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)


    def loss_cross_entropy(self, scores, labels):
        """
        scores: a list of tensors [batch_size, grid_size, grid_size, grid_size, num_classes]
        labels: a list of tensors [batch_size, grid_size, grid_size, grid_size, num_classes]
        """

        with tf.name_scope('loss'):
            loss = 0
            for i in range(cfg.TRAIN.NUM_STEPS):
                score = scores[i]
                label = labels[i]
                cross_entropy = -tf.reduce_sum(label * score, reduction_indices=[4])
                loss += tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(label))
            loss /= cfg.TRAIN.NUM_STEPS
        return loss

    def loss_cross_entropy_single_frame(self, scores, labels):
        """
        scores: a tensor [batch_size, grid_size, grid_size, grid_size, num_classes]
        labels: a tensor [batch_size, grid_size, grid_size, grid_size, num_classes]
        """

        with tf.name_scope('loss'):
            # cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[4])
            cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
            loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels))

        return loss


    def train_model(self, sess, max_iters):
        """Network training loop."""

        if cfg.TRAIN.SINGLE_FRAME:
            # data layer
            data_layer = GtSingleDataLayer(self.roidb, self.imdb.num_classes)
            # classification loss
            scores = self.net.get_output('prob')
            # labels = self.net.get_output('backprojection')[1]
            labels = self.net.get_output('gt_label_2d')
            loss = self.loss_cross_entropy_single_frame(scores, labels)
        else:
            # data layer
            data_layer = GtDataLayer(self.roidb, self.imdb.num_classes)
            # classification loss
            scores = self.net.get_output('outputs')
            labels = self.net.get_output('labels_gt')
            loss = self.loss_cross_entropy(scores, labels)

        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # learning rate
            if iter >= cfg.TRAIN.STEPSIZE:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            else:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            if cfg.TRAIN.SINGLE_FRAME:
                feed_dict={self.net.data: blobs['data_image'], self.net.gt_label_2d: blobs['data_label'], \
                           self.net.depth: blobs['data_depth'], self.net.meta_data: blobs['data_meta_data'], \
                           self.net.gt_label_3d: blobs['data_label_3d']}
            else:
                feed_dict={self.net.data: blobs['data_image'], self.net.gt_label_2d: blobs['data_label'], \
                           self.net.depth: blobs['data_depth'], self.net.meta_data: blobs['data_meta_data'], \
                           self.net.state: blobs['data_state'], self.net.gt_label_3d: blobs['data_label_3d']}
            
            timer.tic()
            loss_cls_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            timer.toc()

            print 'iter: %d / %d, loss_cls: %.4f, lr: %f, time: %.2f' %\
                    (iter+1, max_iters, loss_cls_value, lr.eval(), timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
