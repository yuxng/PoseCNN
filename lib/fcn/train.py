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
from gt_synthesize_layer.layer import GtSynthesizeLayer
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

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.pretrained_ckpt = pretrained_ckpt

        # For checkpoint
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))


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

    def restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes])

        var_name_to_var = {var.name : var for var in tf.global_variables()}
        restore_vars = []
        restored_var_names = set()
        print('Restoring:')
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for var_name, saved_var_name in var_names:
                if 'global_step' in var_name:
                    continue
                if 'Variable' in var_name:
                    continue
                curr_var = var_name_to_var[var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                    print(str(saved_var_name))
                    restored_var_names.add(saved_var_name)
                else:
                    print('Shape mismatch for var', saved_var_name, 'expected', var_shape, 'got', saved_shapes[saved_var_name])
        ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
        if len(ignored_var_names) == 0:
            print('Restored all variables')
        else:
            print('Did not restore:' + '\n\t'.join(ignored_var_names))

        if len(restore_vars) > 0:
            saver = tf.train.Saver(restore_vars)
            saver.restore(session, save_file)
        print('Restored %s' % save_file)


    def train_model(self, sess, train_op, loss, learning_rate, max_iters):
        """Network training loop."""
        # add summary
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        print self.pretrained_ckpt
        if self.pretrained_ckpt is not None:
            print ('Loading pretrained ckpt '
                   'weights from {:s}').format(self.pretrained_ckpt)
            self.restore(sess, self.pretrained_ckpt)

        tf.get_default_graph().finalize()

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            summary, loss_value, lr, _ = sess.run([merged, loss, learning_rate, train_op])
            train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, lr: %.8f, time: %.2f' %\
                    (iter+1, max_iters, loss_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


    def train_model_vertex(self, sess, train_op, loss, loss_cls, loss_vertex, learning_rate, max_iters):
        """Network training loop."""
        # add summary
        # tf.summary.scalar('loss', loss)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        if self.pretrained_ckpt is not None:
            print ('Loading pretrained ckpt '
                   'weights from {:s}').format(self.pretrained_ckpt)
            self.restore(sess, self.pretrained_ckpt)

        tf.get_default_graph().finalize()

        if cfg.TRAIN.SINGLE_FRAME:
            # data layer
            if cfg.TRAIN.SYNTHESIZE:
                data_layer = GtSynthesizeLayer(self.roidb, self.imdb.num_classes, self.imdb._extents, self.imdb.cache_path, self.imdb.name, cfg.CAD, cfg.POSE)
            else:
                data_layer = GtSingleDataLayer(self.roidb, self.imdb.num_classes, self.imdb._extents)
        else:
            # data layer
            data_layer = GtDataLayer(self.roidb, self.imdb.num_classes)

        # tf.train.write_graph(sess.graph_def, self.output_dir, 'model.pbtxt')

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            feed_dict = load_and_enqueue_one(sess, self.net, data_layer)
            timer.toc()
            time_data = timer.diff

            timer.tic()
            loss_value, loss_cls_value, loss_vertex_value, lr, _ = sess.run([loss, loss_cls, loss_vertex, learning_rate, train_op], feed_dict=feed_dict)
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, lr: %.8f, time: %.2f, time data: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, lr, timer.diff, time_data)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


    def train_model_vertex_matching(self, sess, train_op, loss, loss_cls, loss_vertex, loss_pose, loss_matching, learning_rate, max_iters):
        """Network training loop."""
        # add summary
        # tf.summary.scalar('loss', loss)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        if self.pretrained_ckpt is not None:
            print ('Loading pretrained ckpt '
                   'weights from {:s}').format(self.pretrained_ckpt)
            self.restore(sess, self.pretrained_ckpt)

        tf.get_default_graph().finalize()

        # tf.train.write_graph(sess.graph_def, self.output_dir, 'model.pbtxt')

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, loss_matching_value, lr, _ = sess.run([loss, loss_cls, loss_vertex, loss_pose, loss_matching, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_pose: %.4f, loss_match: %.4f, lr: %.8f,  time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, loss_matching_value, lr, timer.diff)

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



def load_and_enqueue_one(sess, net, data_layer):
    
    blobs = data_layer.forward()

    if cfg.INPUT == 'RGBD':
        data_blob = blobs['data_image_color']
        data_p_blob = blobs['data_image_depth']
    elif cfg.INPUT == 'COLOR':
        data_blob = blobs['data_image_color']
    elif cfg.INPUT == 'DEPTH':
        data_blob = blobs['data_image_depth']
    elif cfg.INPUT == 'NORMAL':
        data_blob = blobs['data_image_normal']

    if cfg.TRAIN.SINGLE_FRAME:
        if cfg.INPUT == 'RGBD':
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                               net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                               net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data']}
                else:
                    feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                               net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights']}
            else:
                feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}

        else:
            if cfg.TRAIN.VERTEX_REG:
                if cfg.TRAIN.POSE_REG:
                    feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                               net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                               net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data']}
                else:
                    feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                               net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights']}
            else:
                feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}
    else:
        if cfg.INPUT == 'RGBD':
            feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], \
                       net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                       net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}
        else:
            feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], \
                       net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                       net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}
    # sess.run(net.enqueue_op, feed_dict=feed_dict)
    return feed_dict


def load_and_enqueue(sess, net, data_layer, coord):

    while not coord.should_stop():
        blobs = data_layer.forward()

        if cfg.INPUT == 'RGBD':
            data_blob = blobs['data_image_color']
            data_p_blob = blobs['data_image_depth']
        elif cfg.INPUT == 'COLOR':
            data_blob = blobs['data_image_color']
        elif cfg.INPUT == 'DEPTH':
            data_blob = blobs['data_image_depth']
        elif cfg.INPUT == 'NORMAL':
            data_blob = blobs['data_image_normal']

        if cfg.TRAIN.SINGLE_FRAME:
            if cfg.INPUT == 'RGBD':
                if cfg.TRAIN.VERTEX_REG:
                    if cfg.TRAIN.POSE_REG:
                        feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                                   net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data']}
                    else:
                        feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights']}
                else:
                    feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}

            else:
                if cfg.TRAIN.VERTEX_REG:
                    if cfg.TRAIN.POSE_REG:
                        feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                                   net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data']}
                    else:
                        feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights']}
                else:
                    feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}
        else:
            if cfg.INPUT == 'RGBD':
                feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], \
                           net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                           net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}
            else:
                feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], \
                           net.depth: blobs['data_depth'], net.meta_data: blobs['data_meta_data'], \
                           net.state: blobs['data_state'], net.weights: blobs['data_weights'], net.points: blobs['data_points'], net.keep_prob: 0.5}

        sess.run(net.enqueue_op, feed_dict=feed_dict)

def loss_cross_entropy(scores, labels):
    """
    scores: a list of tensors [batch_size, height, width, num_classes]
    labels: a list of tensors [batch_size, height, width, num_classes]
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
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels))

    return loss


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    if cfg.TRAIN.SINGLE_FRAME:
        # classification loss
        if cfg.NETWORK == 'FCN8VGG':
            scores = network.prob
            labels = network.gt_label_2d_queue
            loss = loss_cross_entropy_single_frame(scores, labels)
        else:
            if cfg.TRAIN.VERTEX_REG:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_2d')
                loss_cls = loss_cross_entropy_single_frame(scores, labels)

                vertex_pred = network.get_output('vertex_pred')
                vertex_targets = network.get_output('vertex_targets')
                vertex_weights = network.get_output('vertex_weights')
                loss_vertex = tf.div( tf.reduce_sum(tf.multiply(vertex_weights, tf.abs(tf.subtract(vertex_pred, vertex_targets)))), tf.reduce_sum(vertex_weights) )

                if cfg.TRAIN.POSE_REG:
                    pose_pred = network.get_output('poses_pred')
                    pose_targets = network.get_output('poses_target')
                    pose_weights = network.get_output('poses_weight')
                    loss_pose = tf.div( tf.reduce_sum(tf.multiply(pose_weights, tf.abs(tf.subtract(pose_pred, pose_targets)))), tf.reduce_sum(pose_weights) )

                    if cfg.TRAIN.MATCHING:
                        loss_matching = network.get_output('matching_loss')[0]
                        loss = loss_cls + loss_vertex + loss_pose + loss_matching
                    else:
                        loss = loss_cls + loss_vertex + loss_pose
                else:
                    loss = loss_cls + loss_vertex
            else:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_2d')
                loss = loss_cross_entropy_single_frame(scores, labels)
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
    
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, pretrained_ckpt=pretrained_ckpt)

        # thread to load data
        '''
        if cfg.TRAIN.SINGLE_FRAME:
            # data layer
            if cfg.TRAIN.SYNTHESIZE:
                data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, cfg.CAD, cfg.POSE)
            else:
                data_layer = GtSingleDataLayer(roidb, imdb.num_classes, imdb._extents)
        else:
            # data layer
            data_layer = GtDataLayer(roidb, imdb.num_classes)

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, network, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, network, data_layer, coord))
            t.start()
        '''

        print 'Solving...'
        if cfg.TRAIN.VERTEX_REG:
            if cfg.TRAIN.MATCHING:
                sw.train_model_vertex_matching(sess, train_op, loss, loss_cls, loss_vertex, loss_pose, loss_matching, learning_rate, max_iters)
            else:
                sw.train_model_vertex(sess, train_op, loss, loss_cls, loss_vertex, learning_rate, max_iters)
        else:
            sw.train_model(sess, train_op, loss, learning_rate, max_iters)
        print 'done solving'

        # sess.run(network.close_queue_op)
        # coord.request_stop()
        # coord.join([t])
