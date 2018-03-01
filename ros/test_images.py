#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from fcn.test import test_net_images
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from utils.blob import im_list_to_blob, pad_im, unpad_im
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp
import numpy as np

import rospy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='pretrained model',
                        default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='shapenet_scene_val', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--kfusion', dest='kfusion',
                        help='run kinect fusion or not',
                        default=False, type=bool)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def _get_image_blob(im, im_depth, meta_data):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    # RGB
    im_orig = im.astype(np.float32, copy=True)
    # mask the color image according to depth
    if cfg.EXP_DIR == 'rgbd_scene':
        I = np.where(im_depth == 0)
        im_orig[I[0], I[1], :] = 0

    processed_ims_rescale = []
    im_scale = cfg.TEST.SCALES_BASE[0]
    im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_rescale.append(im_rescale)

    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # depth
    im_orig = im_depth.astype(np.float32, copy=True)
    im_orig = im_orig / im_orig.max() * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

    processed_ims_depth = []
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_depth.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    blob_depth = im_list_to_blob(processed_ims_depth, 3)
    # blob_normal = im_list_to_blob(processed_ims_normal, 3)
    blob_normal = []

    return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, extents, points, symmetry, num_classes):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)
    im_scale = im_scale_factors[0]
    # construct the meta data
    """
    format of the meta_data
    intrinsic matrix: meta_data[0 ~ 8]
    inverse intrinsic matrix: meta_data[9 ~ 17]
    pose_world2live: meta_data[18 ~ 29]
    pose_live2world: meta_data[30 ~ 41]
    voxel step size: meta_data[42, 43, 44]
    voxel min value: meta_data[45, 46, 47]
    """
    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    # mdata[18:30] = pose_world2live.flatten()
    # mdata[30:42] = pose_live2world.flatten()
    if cfg.FLIP_X:
        mdata[0] = -1 * mdata[0]
        mdata[9] = -1 * mdata[9]
        mdata[11] = -1 * mdata[11]
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    meta_data_blob[0,0,0,:] = mdata

    # use a fake label blob of ones
    height = int(im_depth.shape[0] * im_scale)
    width = int(im_depth.shape[1] * im_scale)
    label_blob = np.ones((1, height, width, num_classes), dtype=np.float32)

    pose_blob = np.zeros((1, 13), dtype=np.float32)
    vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.NETWORK == 'FCN8VGG':
        labels_2d, probs = sess.run([net.label_2d, net.prob], feed_dict=feed_dict)
    else:
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

                # non-maximum suppression
                # keep = nms(rois, 0.5)
                # rois = rois[keep, :]
                # poses_init = poses_init[keep, :]
                # poses_pred = poses_pred[keep, :]
                print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                print rois
                print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        elif cfg.TEST.VERTEX_REG_3D:
            labels_2d, probs, vertex_pred = \
                sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred')])
            rois = []
            poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

    return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses


class ImageListener:

    def __init__(self, sess, network, imdb, meta_data):

        self.sess = sess
        self.net = network
        self.imdb = imdb
        self.meta_data = meta_data
        self.cv_bridge = CvBridge()
        self.count = 0

        # initialize a node
        rospy.init_node("image_listener")
        self.posecnn_pub = rospy.Publisher('posecnn_result', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image, queue_size=2)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

     
    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = 'images/%04d.png' % self.count
        print filename
        self.count += 1
        cv2.imwrite(filename, im)

        # run network
        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(self.sess, self.net, im, depth_cv, self.meta_data, \
            self.imdb._extents, self.imdb._points_all, self.imdb._symmetry, imdb.num_classes)

        im_label = self.imdb.labels_to_image(im, labels)

        # publish
        posecnn_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        posecnn_msg.header.stamp = rospy.Time.now()
        posecnn_msg.header.frame_id = rgb.header.frame_id
        posecnn_msg.encoding = 'rgb8'
        self.posecnn_pub.publish(posecnn_msg)

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)

    # construct meta data
    K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 1000.0})
    print meta_data

    cfg.GPU_ID = args.gpu_id
    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print device_name

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    if cfg.NETWORK == 'FCN8VGG':
        path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
        cfg.TRAIN.MODEL_PATH = path
    cfg.TRAIN.TRAINABLE = False

    cfg.RIG = args.rig_name
    cfg.CAD = args.cad_name
    cfg.POSE = args.pose_name
    cfg.BACKGROUND = args.background_name
    cfg.IS_TRAIN = False

    from networks.factory import get_network
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    # start a session
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)

    # image listener
    listener = ImageListener(sess, network, imdb, meta_data)
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
