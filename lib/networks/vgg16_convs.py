import tensorflow as tf
from networks.network import Network

class vgg16_convs(Network):
    def __init__(self, input_format, num_classes, num_units, scales, threshold_label, vote_threshold, vertex_reg_2d=False, vertex_reg_3d=False, pose_reg=False, adaptation=False, trainable=True, is_train=True):
        self.inputs = []
        self.input_format = input_format
        self.num_classes = num_classes
        self.num_units = num_units
        self.scale = 1.0
        self.threshold_label = threshold_label
        self.vertex_reg_2d = vertex_reg_2d
        self.vertex_reg_3d = vertex_reg_3d
        self.vertex_reg = vertex_reg_2d or vertex_reg_3d
        self.pose_reg = pose_reg
        self.adaptation = adaptation
        self.trainable = trainable
        # if vote_threshold < 0, only detect single instance (default). 
        # Otherwise, multiple instances are detected if hough voting score larger than the threshold
        if is_train:
            self.is_train = 1
            self.skip_pixels = 10
            self.vote_threshold = vote_threshold
            self.vote_percentage = 0.02
        else:
            self.is_train = 0
            self.skip_pixels = 10
            self.vote_threshold = vote_threshold
            self.vote_percentage = 0.02

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        if input_format == 'RGBD':
            self.data_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.gt_label_2d = tf.placeholder(tf.int32, shape=[None, None, None])
        self.keep_prob = tf.placeholder(tf.float32)
        if self.vertex_reg:
            self.vertex_targets = tf.placeholder(tf.float32, shape=[None, None, None, 3 * num_classes])
            self.vertex_weights = tf.placeholder(tf.float32, shape=[None, None, None, 3 * num_classes])
            self.poses = tf.placeholder(tf.float32, shape=[None, 13])
            self.extents = tf.placeholder(tf.float32, shape=[num_classes, 3])
            self.meta_data = tf.placeholder(tf.float32, shape=[None, 1, 1, 48])
            self.points = tf.placeholder(tf.float32, shape=[num_classes, None, 3])
            self.symmetry = tf.placeholder(tf.float32, shape=[num_classes])

        # define a queue
        queue_size = 25
        if input_format == 'RGBD':
            if self.vertex_reg:
                q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
                self.enqueue_op = q.enqueue([self.data, self.data_p, self.gt_label_2d, self.keep_prob, \
                                             self.vertex_targets, self.vertex_weights, self.poses, \
                                             self.extents, self.meta_data, self.points, self.symmetry])
                data, data_p, gt_label_2d, self.keep_prob_queue, vertex_targets, vertex_weights, poses, extents, meta_data, points, symmetry = q.dequeue()
                self.layers = dict({'data': data, 'data_p': data_p, 'gt_label_2d': gt_label_2d, 'vertex_targets': vertex_targets, \
                                    'vertex_weights': vertex_weights, 'poses': poses, 'extents': extents, \
                                    'meta_data': meta_data, 'points': points, 'symmetry': symmetry})
            else:
                q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32, tf.float32])
                self.enqueue_op = q.enqueue([self.data, self.data_p, self.gt_label_2d, self.keep_prob])
                data, data_p, gt_label_2d, self.keep_prob_queue = q.dequeue()
                self.layers = dict({'data': data, 'data_p': data_p, 'gt_label_2d': gt_label_2d})
        else:
            if self.vertex_reg:
                q = tf.FIFOQueue(queue_size, [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
                self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.keep_prob, self.vertex_targets, self.vertex_weights, self.poses, self.extents, self.meta_data, self.points, self.symmetry])
                data, gt_label_2d, self.keep_prob_queue, vertex_targets, vertex_weights, poses, extents, meta_data, points, symmetry = q.dequeue()
                self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d, 'vertex_targets': vertex_targets, 'vertex_weights': vertex_weights, 
                                    'poses': poses, 'extents': extents, 'meta_data': meta_data, 'points': points, 'symmetry': symmetry})
            else:
                q = tf.FIFOQueue(queue_size, [tf.float32, tf.int32, tf.float32])
                self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.keep_prob])
                data, gt_label_2d, self.keep_prob_queue = q.dequeue()
                self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d})
        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        self.queue_size = q.size()

        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=self.trainable)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=self.trainable)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=self.trainable))

        if self.input_format == 'RGBD':
            (self.feed('data_p')
                 .conv(3, 3, 64, 1, 1, name='conv1_1_p', c_i=3, trainable=self.trainable)
                 .conv(3, 3, 64, 1, 1, name='conv1_2_p', c_i=64, trainable=self.trainable)
                 .max_pool(2, 2, 2, 2, name='pool1_p')
                 .conv(3, 3, 128, 1, 1, name='conv2_1_p', c_i=64, trainable=self.trainable)
                 .conv(3, 3, 128, 1, 1, name='conv2_2_p', c_i=128, trainable=self.trainable)
                 .max_pool(2, 2, 2, 2, name='pool2_p')
                 .conv(3, 3, 256, 1, 1, name='conv3_1_p', c_i=128, trainable=self.trainable)
                 .conv(3, 3, 256, 1, 1, name='conv3_2_p', c_i=256, trainable=self.trainable)
                 .conv(3, 3, 256, 1, 1, name='conv3_3_p', c_i=256, trainable=self.trainable)
                 .max_pool(2, 2, 2, 2, name='pool3_p')
                 .conv(3, 3, 512, 1, 1, name='conv4_1_p', c_i=256, trainable=self.trainable)
                 .conv(3, 3, 512, 1, 1, name='conv4_2_p', c_i=512, trainable=self.trainable)
                 .conv(3, 3, 512, 1, 1, name='conv4_3_p', c_i=512, trainable=self.trainable)
                 .max_pool(2, 2, 2, 2, name='pool4_p')
                 .conv(3, 3, 512, 1, 1, name='conv5_1_p', c_i=512, trainable=self.trainable)
                 .conv(3, 3, 512, 1, 1, name='conv5_2_p', c_i=512, trainable=self.trainable)
                 .conv(3, 3, 512, 1, 1, name='conv5_3_p', c_i=512, trainable=self.trainable))

            (self.feed('conv5_3', 'conv5_3_p')
                 .concat(3, name='concat_conv5')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=1024)
                 .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False))

            (self.feed('conv4_3', 'conv4_3_p')
                 .concat(3, name='concat_conv4')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=1024))
        else:
            (self.feed('conv5_3')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512)
                 .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False))

            (self.feed('conv4_3')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512))

        (self.feed('score_conv4', 'upscore_conv5')
             .add(name='add_score')
             .dropout(self.keep_prob_queue, name='dropout')
             .deconv(int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False))

        (self.feed('upscore')
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=self.num_units)
             .log_softmax_high_dimension(self.num_classes, name='prob'))

        (self.feed('score')
             .softmax_high_dimension(self.num_classes, name='prob_normalized')
             .argmax_2d(name='label_2d'))

        (self.feed('prob_normalized', 'gt_label_2d')
             .hard_label(threshold=self.threshold_label, name='gt_label_weight'))

        if self.vertex_reg:
            (self.feed('conv5_3')
                 .conv(1, 1, 128, 1, 1, name='score_conv5_vertex', relu=False, c_i=512)
                 .deconv(4, 4, 128, 2, 2, name='upscore_conv5_vertex', trainable=False))

            (self.feed('conv4_3')
                 .conv(1, 1, 128, 1, 1, name='score_conv4_vertex', relu=False, c_i=512))
            
            (self.feed('score_conv4_vertex', 'upscore_conv5_vertex')
                 .add(name='add_score_vertex')
                 .dropout(self.keep_prob_queue, name='dropout_vertex')
                 .deconv(int(16*self.scale), int(16*self.scale), 128, int(8*self.scale), int(8*self.scale), name='upscore_vertex', trainable=False)
                 .conv(1, 1, 3 * self.num_classes, 1, 1, name='vertex_pred', relu=False, c_i=128))

            if self.vertex_reg_2d:

                (self.feed('label_2d', 'vertex_pred', 'extents', 'meta_data', 'poses')
                     .hough_voting_gpu(self.is_train, self.vote_threshold, self.vote_percentage, self.skip_pixels, name='hough'))

                self.layers['rois'] = self.get_output('hough')[0]
                self.layers['poses_init'] = self.get_output('hough')[1]
                self.layers['poses_target'] = self.get_output('hough')[2]
                self.layers['poses_weight'] = self.get_output('hough')[3]
                
                if self.pose_reg:
                    # roi pooling without masking
                    (self.feed('conv5_3', 'rois')
                         .roi_pool(7, 7, 1.0 / 16.0, 0, name='pool5'))
                         #.crop_pool_new(16.0, pool_size=7, name='pool5'))
                         
                    (self.feed('conv4_3', 'rois')
                         .roi_pool(7, 7, 1.0 / 8.0, 0, name='pool4'))
                         #.crop_pool_new(8.0, pool_size=7, name='pool4'))


                    (self.feed('pool5', 'pool4')
                         .add(name='pool_score')
                         .fc(4096, height=7, width=7, channel=512, name='fc6')
                         .dropout(self.keep_prob_queue, name='drop6')
                         .fc(4096, num_in=4096, name='fc7')
                         .dropout(self.keep_prob_queue, name='drop7')
                         .fc(4 * self.num_classes, relu=False, name='fc8')
                         .tanh(name='poses_tanh'))

                    (self.feed('poses_tanh', 'poses_weight')
                         .multiply(name='poses_mul')
                         .l2_normalize(dim=1, name='poses_pred'))

                    (self.feed('poses_pred', 'poses_target', 'poses_weight', 'points', 'symmetry')
                         .average_distance_loss(margin=0.01, name='loss_pose'))

                    # domain adaptation
                    if self.adaptation:
                        self.layers['label_domain'] = self.get_output('hough')[4]

                        (self.feed('pool_score')
                             .gradient_reversal(0.01, name='greversal')
                             .fc(256, height=7, width=7, channel=512, name='fc9')
                             .dropout(self.keep_prob_queue, name='drop9') 
                             .fc(2, name='domain_score')
                             .softmax(-1, name='domain_prob')
                             .argmax(-1, name='domain_label'))
