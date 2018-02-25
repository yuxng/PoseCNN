import tensorflow as tf
from networks.network import Network

class vgg16_det(Network):
    def __init__(self, input_format, num_classes, feature_stride, anchor_scales, anchor_ratios, trainable=True, is_train=True):
        self.inputs = []
        self.input_format = input_format
        self.num_classes = num_classes
        self.feature_stride = feature_stride
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.anchor_ratios = anchor_ratios
        self.num_ratios = len(anchor_ratios)
        self.num_anchors = self.num_scales * self.num_ratios
        if is_train:
            self.is_train = 1
            self.mode = 'TRAIN'
        else:
            self.is_train = 0
            self.mode = 'TEST'

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        if input_format == 'RGBD':
            self.data_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.poses = tf.placeholder(tf.float32, shape=[None, 13])
        self.points = tf.placeholder(tf.float32, shape=[num_classes, None, 3])
        self.symmetry = tf.placeholder(tf.float32, shape=[num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        queue_size = 25
        if input_format == 'RGBD':
            q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.data_p, self.im_info, self.gt_boxes, self.poses, self.points, self.symmetry, self.keep_prob])
            data, data_p, im_info, gt_boxes, poses, points, symmetry, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': data, 'data_p': data_p, 'im_info': im_info, 'gt_boxes': gt_boxes, \
                                'poses': poses, 'points': points, 'symmetry': symmetry})
        else:
            q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.im_info, self.gt_boxes, self.poses, self.points, self.symmetry, self.keep_prob])
            data, im_info, gt_boxes, poses, points, symmetry, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': data, 'im_info': im_info, 'gt_boxes': gt_boxes, 'poses': poses, 'points': points, 'symmetry': symmetry})
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512)
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512))

        if self.input_format == 'RGBD':
            (self.feed('data_p')
                 .conv(3, 3, 64, 1, 1, name='conv1_1_p', c_i=3)
                 .conv(3, 3, 64, 1, 1, name='conv1_2_p', c_i=64)
                 .max_pool(2, 2, 2, 2, name='pool1_p')
                 .conv(3, 3, 128, 1, 1, name='conv2_1_p', c_i=64)
                 .conv(3, 3, 128, 1, 1, name='conv2_2_p', c_i=128)
                 .max_pool(2, 2, 2, 2, name='pool2_p')
                 .conv(3, 3, 256, 1, 1, name='conv3_1_p', c_i=128)
                 .conv(3, 3, 256, 1, 1, name='conv3_2_p', c_i=256)
                 .conv(3, 3, 256, 1, 1, name='conv3_3_p', c_i=256)
                 .max_pool(2, 2, 2, 2, name='pool3_p')
                 .conv(3, 3, 512, 1, 1, name='conv4_1_p', c_i=256)
                 .conv(3, 3, 512, 1, 1, name='conv4_2_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv4_3_p', c_i=512)
                 .max_pool(2, 2, 2, 2, name='pool4_p')
                 .conv(3, 3, 512, 1, 1, name='conv5_1_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_2_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_3_p', c_i=512))

            (self.feed('conv5_3', 'conv5_3_p')
                 .concat(3, name='concat_conv5')
                 .conv(3, 3, 512, 1, 1, name='conv_rpn', c_i=1024))
        else:
            (self.feed('conv5_3')
                 .conv(3, 3, 512, 1, 1, name='conv_rpn', c_i=512))

        (self.feed('conv_rpn')
             .conv(1, 1, self.num_anchors * 2, 1, 1, name='rpn_cls_score', c_i=512)
             .reshape_score(2, name='rpn_cls_score_reshape')
             .softmax_high_dimension(2, name='rpn_cls_prob_reshape')
             .reshape_score(self.num_anchors * 2, name='rpn_cls_prob'))

        (self.feed('conv_rpn')
             .conv(1, 1, self.num_anchors * 4, 1, 1, name='rpn_bbox_pred', c_i=512))

        # compute anchors
        (self.feed('im_info')
             .compute_anchors(self.feature_stride, self.anchor_scales, self.anchor_ratios, name='anchors'))

        # compute rpn targets
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'anchors')
             .compute_anchor_targets(self.num_anchors, name='anchor_targets'))

        self.layers['rpn_labels'] = self.get_output('anchor_targets')[0]
        self.layers['rpn_bbox_targets'] = self.get_output('anchor_targets')[1]
        self.layers['rpn_bbox_inside_weights'] = self.get_output('anchor_targets')[2]
        self.layers['rpn_bbox_outside_weights'] = self.get_output('anchor_targets')[3]

        # compute region proposals
        (self.feed('rpn_cls_prob', 'rpn_bbox_pred', 'im_info', 'anchors')
             .compute_proposals(self.feature_stride, self.num_anchors, self.mode, name='proposals'))

        self.layers['rois'] = self.get_output('proposals')[0]
        self.layers['rpn_scores'] = self.get_output('proposals')[1]

        # compute proposal targets
        (self.feed('rois', 'rpn_scores', 'gt_boxes', 'poses')
             .compute_proposal_targets(self.num_classes, name='proposal_targets'))

        if self.is_train:
            self.layers['rois_target'] = self.get_output('proposal_targets')[0]
        else:
            self.layers['rois_target'] = self.layers['rois']
        self.layers['rpn_scores_target'] = self.get_output('proposal_targets')[1]
        self.layers['labels'] = self.get_output('proposal_targets')[2]
        self.layers['bbox_targets'] = self.get_output('proposal_targets')[3]
        self.layers['bbox_inside_weights'] = self.get_output('proposal_targets')[4]
        self.layers['bbox_outside_weights'] = self.get_output('proposal_targets')[5]
        self.layers['poses_target'] = self.get_output('proposal_targets')[6]
        self.layers['poses_weight'] = self.get_output('proposal_targets')[7]

        # classify rois
        (self.feed('conv5_3', 'rois_target')
             .crop_pool(self.feature_stride, pool_size=7, name='pool5')
             .fc(4096, height=7, width=7, channel=512, name='fc6')
             .dropout(self.keep_prob_queue, name='drop6')
             .fc(4096, num_in=4096, name='fc7')
             .dropout(self.keep_prob_queue, name='drop7') 
             .fc(self.num_classes, name='cls_score')
             .softmax(-1, name='cls_prob'))

        # bounding box regression
        (self.feed('drop7')
             .fc(4 * self.num_classes, relu=False, name='bbox_pred'))

        # pose regression
        (self.feed('drop7')
             .fc(4 * self.num_classes, relu=False, name='poses_pred_unnormalized')
             .tanh(name='poses_tanh'))

        (self.feed('poses_tanh', 'poses_weight')
             .multiply(name='poses_mul')
             .l2_normalize(dim=1, name='poses_pred'))

        (self.feed('poses_pred', 'poses_target', 'poses_weight', 'points', 'symmetry')
             .average_distance_loss(name='loss_pose'))
