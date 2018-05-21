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
import math

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
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=12)


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename, write_meta_graph=False)
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


    def train_model(self, sess, train_op, loss, learning_rate, max_iters, data_layer):
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

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

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

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


    def train_model_vertex(self, sess, train_op, loss, loss_cls, loss_vertex, loss_regu, learning_rate, max_iters, data_layer):
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

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

        # tf.train.write_graph(sess.graph_def, self.output_dir, 'model.pbtxt')

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):

            timer.tic()
            loss_value, loss_cls_value, loss_vertex_value, loss_regu_value, lr, _ = sess.run([loss, loss_cls, loss_vertex, loss_regu, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_regu: %.12f, lr: %.8f, time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, loss_regu_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


    def train_model_vertex_pose(self, sess, train_op, loss, loss_cls, loss_vertex, loss_pose, learning_rate, max_iters, data_layer):
        """Network training loop."""
        # add summary
        # tf.summary.scalar('loss', loss)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

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
            loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, lr, _ = sess.run([loss, loss_cls, loss_vertex, loss_pose, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_pose: %.4f, lr: %.8f,  time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


    def train_model_vertex_pose_adapt(self, sess, train_op, loss, loss_cls, loss_vertex, loss_pose, \
        loss_domain, label_domain, domain_label, learning_rate, max_iters, data_layer):
        """Network training loop."""

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

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

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):

            timer.tic()
            loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, loss_domain_value, label_domain_value, domain_label_value, lr, _ = sess.run([loss, loss_cls, loss_vertex, loss_pose, loss_domain, label_domain, domain_label, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_pose: %.4f, loss_domain: %.4f, lr: %.8f,  time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, loss_domain_value, lr, timer.diff)
            print label_domain_value
            print domain_label_value

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


    def train_model_det(self, sess, train_op, loss, loss_rpn_cls, loss_rpn_box, loss_cls, loss_box, loss_pose, learning_rate, max_iters, data_layer):
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

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):

            timer.tic()
            loss_value, loss_rpn_cls_value, loss_rpn_box_value, loss_cls_value, loss_box_value, loss_pose_value, lr, _ \
                = sess.run([loss, loss_rpn_cls, loss_rpn_box, loss_cls, loss_box, loss_pose, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_rpn_cls: %.4f, loss_rpn_box: %.4f, loss_cls: %.4f, loss_box: %.4f, loss_pose: %.4f, lr: %.8f, time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_rpn_cls_value, loss_rpn_box_value, loss_cls_value, loss_box_value, loss_pose_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb


def load_and_enqueue(sess, net, data_layer, coord):

    iter = 0
    while not coord.should_stop():
        blobs = data_layer.forward(iter)
        iter += 1

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
            if cfg.TRAIN.SEGMENTATION:
                if cfg.INPUT == 'RGBD':
                    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
                        feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                                   net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data']}
                    else:
                        feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}
                else:
                    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
                        feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                                   net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data'], \
                                   net.points: blobs['data_points'], net.symmetry: blobs['data_symmetry']}
                    else:
                        feed_dict={net.data: data_blob, net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5}
            else:
                if cfg.INPUT == 'RGBD':
                    feed_dict={net.data: data_blob, net.data_p: data_p_blob, net.im_info: blobs['data_im_info'], \
                               net.gt_boxes: blobs['data_gt_boxes'], net.poses: blobs['data_pose'], \
                               net.points: blobs['data_points'], net.symmetry: blobs['data_symmetry'], net.keep_prob: 0.5}
                else:
                    feed_dict={net.data: data_blob, net.im_info: blobs['data_im_info'], \
                               net.gt_boxes: blobs['data_gt_boxes'], net.poses: blobs['data_pose'], \
                               net.points: blobs['data_points'], net.symmetry: blobs['data_symmetry'], net.keep_prob: 0.5}
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
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels)+1e-10)

    return loss


def loss_quaternion(pose_pred, pose_targets, pose_weights):

    with tf.name_scope('loss'):
        distances = 1 - tf.square( tf.reduce_sum(tf.multiply(pose_pred, pose_targets), reduction_indices=[1]) )
        weights = tf.reduce_mean(pose_weights, reduction_indices=[1])
        loss = tf.div( tf.reduce_sum(tf.multiply(weights, distances)), tf.reduce_sum(weights)+1e-10 )

    return loss


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
    if cfg.TRAIN.SINGLE_FRAME:
        # classification loss
        if cfg.NETWORK == 'FCN8VGG':
            scores = network.prob
            labels = network.gt_label_2d_queue
            loss = loss_cross_entropy_single_frame(scores, labels) + loss_regu
        else:
            if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_weight')
                loss_cls = loss_cross_entropy_single_frame(scores, labels)

                vertex_pred = network.get_output('vertex_pred')
                vertex_targets = network.get_output('vertex_targets')
                vertex_weights = network.get_output('vertex_weights')
                # loss_vertex = tf.div( tf.reduce_sum(tf.multiply(vertex_weights, tf.abs(tf.subtract(vertex_pred, vertex_targets)))), tf.reduce_sum(vertex_weights) + 1e-10 )
                loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

                if cfg.TRAIN.POSE_REG:
                    # pose_pred = network.get_output('poses_pred')
                    # pose_targets = network.get_output('poses_target')
                    # pose_weights = network.get_output('poses_weight')
                    # loss_pose = cfg.TRAIN.POSE_W * tf.div( tf.reduce_sum(tf.multiply(pose_weights, tf.abs(tf.subtract(pose_pred, pose_targets)))), tf.reduce_sum(pose_weights) )
                    # loss_pose = cfg.TRAIN.POSE_W * loss_quaternion(pose_pred, pose_targets, pose_weights)
                    loss_pose = cfg.TRAIN.POSE_W * network.get_output('loss_pose')[0]

                    if cfg.TRAIN.ADAPT:
                        domain_score = network.get_output("domain_score")
                        domain_label = network.get_output("domain_label")
                        label_domain = network.get_output("label_domain")
                        loss_domain = cfg.TRAIN.ADAPT_WEIGHT * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=domain_score, labels=label_domain))
                        loss = loss_cls + loss_vertex + loss_pose + loss_domain + loss_regu 
                    else:
                        loss = loss_cls + loss_vertex + loss_pose + loss_regu
                else:
                    loss = loss_cls + loss_vertex + loss_regu
            else:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_weight')
                loss = loss_cross_entropy_single_frame(scores, labels) + loss_regu
    else:
        # classification loss
        scores = network.get_output('outputs')
        labels = network.get_output('labels_gt_2d')
        loss = loss_cross_entropy(scores, labels) + loss_regu

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
    
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.85
    #config.gpu_options.allow_growth = True
    #with tf.Session(config=config) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # data layer
        if cfg.TRAIN.SINGLE_FRAME:
            data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry, imdb.cache_path, imdb.name, imdb.data_queue, cfg.CAD, cfg.POSE)
        else:
            data_layer = GtDataLayer(roidb, imdb.num_classes)

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, pretrained_ckpt=pretrained_ckpt)

        print 'Solving...'
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            if cfg.TRAIN.POSE_REG:
                if cfg.TRAIN.ADAPT:
                    sw.train_model_vertex_pose_adapt(sess, train_op, loss, loss_cls, loss_vertex, loss_pose, \
                        loss_domain, label_domain, domain_label, learning_rate, max_iters, data_layer)
                else:
                    sw.train_model_vertex_pose(sess, train_op, loss, loss_cls, loss_vertex, loss_pose, learning_rate, max_iters, data_layer)
            else:
                sw.train_model_vertex(sess, train_op, loss, loss_cls, loss_vertex, loss_regu, learning_rate, max_iters, data_layer)
        else:
            sw.train_model(sess, train_op, loss, learning_rate, max_iters, data_layer)
        print 'done solving'

def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = tf.multiply(vertex_weights, vertex_diff)
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = tf.div( tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10 )
    return loss


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box


def train_net_det(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')

    # RPN, class loss
    rpn_cls_score = tf.reshape(network.get_output('rpn_cls_score_reshape'), [-1, 2])
    rpn_label = tf.reshape(network.get_output('rpn_labels'), [-1])
    rpn_select = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
    loss_rpn_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # RPN, bbox loss
    rpn_bbox_pred = network.get_output('rpn_bbox_pred')
    rpn_bbox_targets = network.get_output('rpn_bbox_targets')
    rpn_bbox_inside_weights = network.get_output('rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = network.get_output('rpn_bbox_outside_weights')
    loss_rpn_box = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

    # RCNN, class loss
    cls_score = network.get_output("cls_score")
    label = tf.reshape(network.get_output("labels"), [-1])
    loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

    # RCNN, bbox loss
    bbox_pred = network.get_output('bbox_pred')
    bbox_targets = network.get_output('bbox_targets')
    bbox_inside_weights = network.get_output('bbox_inside_weights')
    bbox_outside_weights = network.get_output('bbox_outside_weights')
    loss_box = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # pose regression loss
    loss_pose = network.get_output('loss_pose')[0]

    # add losses
    loss = loss_rpn_cls + loss_rpn_box + loss_cls + loss_box + loss_pose + loss_regu

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
    
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.85
    #config.gpu_options.allow_growth = True
    #with tf.Session(config=config) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, pretrained_ckpt=pretrained_ckpt)

        # thread to load data
        data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry, imdb.cache_path, imdb.name, cfg.CAD, cfg.POSE)

        print 'Solving...'
        sw.train_model_det(sess, train_op, loss, loss_rpn_cls, loss_rpn_box, loss_cls, loss_box, loss_pose, learning_rate, max_iters, data_layer)
        print 'done solving'
