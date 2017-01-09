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
import threading

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


    def train_model(self, sess, train_op, loss_cls, loss_metric, learning_rate, max_iters):
        """Network training loop."""

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        tf.get_default_graph().finalize()

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            loss_cls_value, loss_metric_value, lr, _ = sess.run([loss_cls, loss_metric, learning_rate, train_op])
            timer.toc()
            
            print 'iter: %d / %d, loss_cls: %.4f, loss_metric: %.4f, lr: %f, time: %.2f' %\
                    (iter+1, max_iters, loss_cls_value, loss_metric_value, lr, timer.diff)

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


def load_and_enqueue(sess, net, roidb, num_classes, coord):
    if cfg.TRAIN.SINGLE_FRAME:
        # data layer
        data_layer = GtSingleDataLayer(roidb, num_classes)
    else:
        # data layer
        data_layer = GtDataLayer(roidb, num_classes)

    while not coord.should_stop():
        blobs = data_layer.forward()

        if cfg.TRAIN.SINGLE_FRAME:
            feed_dict={net.data: blobs['data_image'], net.gt_label_2d: blobs['data_label'], \
                       net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data']}
        else:
            feed_dict={net.data: blobs['data_image'], net.gt_label_2d: blobs['data_label'], \
                       net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                       net.state: blobs['data_state'], net.points: blobs['data_points']}

        sess.run(net.enqueue_op, feed_dict=feed_dict)

def loss_cross_entropy(scores, labels):
    """
    scores: a list of tensors [batch_size, grid_size, grid_size, grid_size, num_classes]
    labels: a list of tensors [batch_size, grid_size, grid_size, grid_size, num_classes]
    """

    with tf.name_scope('loss'):
        loss = 0
        for i in range(cfg.TRAIN.NUM_STEPS):
            score = scores[i]
            label = labels[i]
            cross_entropy = -tf.reduce_sum(label * score, reduction_indices=[3])
            loss += tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(label))
        loss /= cfg.TRAIN.NUM_STEPS
    return loss

def loss_cross_entropy_single_frame(scores, labels):
    """
    scores: a tensor [batch_size, grid_size, grid_size, grid_size, num_classes]
    labels: a tensor [batch_size, grid_size, grid_size, grid_size, num_classes]
    """

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels))

    return loss


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    if cfg.TRAIN.SINGLE_FRAME:
        # classification loss
        scores = network.get_output('prob')
        labels = network.get_output('gt_label_2d')
        loss_metric = network.get_output('triplet')[0]
        loss_cls = loss_cross_entropy_single_frame(scores, labels)
        loss = 10 * loss_cls + loss_metric
    else:
        # classification loss
        scores = network.get_output('outputs')
        labels = network.get_output('labels_gt_2d')
        loss = loss_cross_entropy(scores, labels)

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        # thread to load data
        coord = tf.train.Coordinator()
        t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
        t.start()

        print 'Solving...'
        sw.train_model(sess, train_op, loss_cls, loss_metric, learning_rate, max_iters)
        print 'done solving'

        sess.run(network.close_queue_op)
        coord.request_stop()
        coord.join([t])
