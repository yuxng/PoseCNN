import tensorflow as tf
from networks.network import Network

class vgg16(Network):
    def __init__(self, input_format, num_steps, num_classes, num_units, scales, trainable=True):
        self.inputs = []
        self.input_format = input_format
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_units = num_units
        self.scale = 1 / scales[0]

        self.data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 3])
        if input_format == 'RGBD':
            self.data_p = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 3])

        self.gt_label_2d = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 48])
        self.state = tf.placeholder(tf.float32, [None, None, None, self.num_units])
        self.weights = tf.placeholder(tf.float32, [None, None, None, self.num_units])
        self.points = tf.placeholder(tf.float32, [None, None, None, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        if input_format == 'RGBD':
            q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.data_p, self.gt_label_2d, self.depth, self.meta_data, self.state, self.weights, self.points, self.keep_prob])
            self.data_queue, self.data_p_queue, self.gt_label_2d_queue, self.depth_queue, self.meta_data_queue, self.state_queue, self.weights_queue, self.points_queue, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': [], 'data_p': [], 'gt_label_2d': [], 'depth': [], 'meta_data': [], 'state': [], 'weights': [], 'points': []})
        else:
            q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.depth, self.meta_data, self.state, self.weights, self.points, self.keep_prob])
            self.data_queue, self.gt_label_2d_queue, self.depth_queue, self.meta_data_queue, self.state_queue, self.weights_queue, self.points_queue, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': [], 'gt_label_2d': [], 'depth': [], 'meta_data': [], 'state': [], 'weights': [], 'points': []})

        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        self.trainable = trainable
        self.setup()

    def setup(self):
        input_data = tf.unstack(self.data_queue, self.num_steps)
        if self.input_format == 'RGBD':
            input_data_p = tf.unstack(self.data_p_queue, self.num_steps)
        input_label_2d = tf.unstack(self.gt_label_2d_queue, self.num_steps)
        input_depth = tf.unstack(self.depth_queue, self.num_steps)
        input_meta_data = tf.unstack(self.meta_data_queue, self.num_steps)
        input_state = self.state_queue
        input_weights = self.weights_queue
        input_points = self.points_queue
        outputs = []
        probs = []
        labels_gt_2d = []
        labels_pred_2d = []
        
        for i in range(self.num_steps):
            # set inputs
            self.layers['data'] = input_data[i]
            if self.input_format == 'RGBD':
                self.layers['data_p'] = input_data_p[i]
            self.layers['gt_label_2d'] = input_label_2d[i]
            self.layers['depth'] = input_depth[i]
            self.layers['meta_data'] = input_meta_data[i]
            self.layers['state'] = input_state
            self.layers['weights'] = input_weights
            self.layers['points'] = input_points
            if i == 0:
                reuse = None
            else:
                reuse = True

            (self.feed('data')
                 .conv(3, 3, 64, 1, 1, name='conv1_1', reuse=reuse, c_i=3)
                 .conv(3, 3, 64, 1, 1, name='conv1_2', reuse=reuse, c_i=64)
                 .max_pool(2, 2, 2, 2, name='pool1')
                 .conv(3, 3, 128, 1, 1, name='conv2_1', reuse=reuse, c_i=64)
                 .conv(3, 3, 128, 1, 1, name='conv2_2', reuse=reuse, c_i=128)
                 .max_pool(2, 2, 2, 2, name='pool2')
                 .conv(3, 3, 256, 1, 1, name='conv3_1', reuse=reuse, c_i=128)
                 .conv(3, 3, 256, 1, 1, name='conv3_2', reuse=reuse, c_i=256)
                 .conv(3, 3, 256, 1, 1, name='conv3_3', reuse=reuse, c_i=256)
                 .max_pool(2, 2, 2, 2, name='pool3')
                 .conv(3, 3, 512, 1, 1, name='conv4_1', reuse=reuse, c_i=256)
                 .conv(3, 3, 512, 1, 1, name='conv4_2', reuse=reuse, c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv4_3', reuse=reuse, c_i=512)
                 .max_pool(2, 2, 2, 2, name='pool4')
                 .conv(3, 3, 512, 1, 1, name='conv5_1', reuse=reuse, c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_2', reuse=reuse, c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_3', reuse=reuse, c_i=512))

            if self.input_format == 'RGBD': 
                (self.feed('data_p')
                     .conv(3, 3, 64, 1, 1, name='conv1_1_p', reuse=reuse, c_i=3)
                     .conv(3, 3, 64, 1, 1, name='conv1_2_p', reuse=reuse, c_i=64)
                     .max_pool(2, 2, 2, 2, name='pool1_p')
                     .conv(3, 3, 128, 1, 1, name='conv2_1_p', reuse=reuse, c_i=64)
                     .conv(3, 3, 128, 1, 1, name='conv2_2_p', reuse=reuse, c_i=128)
                     .max_pool(2, 2, 2, 2, name='pool2_p')
                     .conv(3, 3, 256, 1, 1, name='conv3_1_p', reuse=reuse, c_i=128)
                     .conv(3, 3, 256, 1, 1, name='conv3_2_p', reuse=reuse, c_i=256)
                     .conv(3, 3, 256, 1, 1, name='conv3_3_p', reuse=reuse, c_i=256)
                     .max_pool(2, 2, 2, 2, name='pool3_p')
                     .conv(3, 3, 512, 1, 1, name='conv4_1_p', reuse=reuse, c_i=256)
                     .conv(3, 3, 512, 1, 1, name='conv4_2_p', reuse=reuse, c_i=512)
                     .conv(3, 3, 512, 1, 1, name='conv4_3_p', reuse=reuse, c_i=512)
                     .max_pool(2, 2, 2, 2, name='pool4_p')
                     .conv(3, 3, 512, 1, 1, name='conv5_1_p', reuse=reuse, c_i=512)
                     .conv(3, 3, 512, 1, 1, name='conv5_2_p', reuse=reuse, c_i=512)
                     .conv(3, 3, 512, 1, 1, name='conv5_3_p', reuse=reuse, c_i=512))

                (self.feed('conv5_3', 'conv5_3_p')
                     .concat(3, name='concat_conv5')
                     .conv(1, 1, self.num_units, 1, 1, name='score_conv5', reuse=reuse, c_i=1024)
                     .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', reuse=reuse, trainable=False))

                (self.feed('conv4_3', 'conv4_3_p')
                     .concat(3, name='concat_conv4')
                     .conv(1, 1, self.num_units, 1, 1, name='score_conv4', reuse=reuse, c_i=1024))
            else:
                (self.feed('conv5_3')
                     .conv(1, 1, self.num_units, 1, 1, name='score_conv5', reuse=reuse, c_i=512)
                     .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', reuse=reuse, trainable=False))

                (self.feed('conv4_3')
                     .conv(1, 1, self.num_units, 1, 1, name='score_conv4', reuse=reuse, c_i=512))

            (self.feed('score_conv4', 'upscore_conv5')
                 .add(name='add_score')
                 .deconv(int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', reuse=reuse, trainable=False))

            (self.feed('state', 'weights', 'points', 'depth', 'meta_data')
                 .compute_flow(3, 0.02, 50, name='flow'))

            # self.layers['flow'] = (self.get_output('state'), self.get_output('weights'), self.get_output('points'))

            (self.feed('upscore', 'flow')
                 .rnn_gru2d(self.num_units, self.num_units, name='gru2d', reuse=reuse)
                 .conv(1, 1, self.num_classes, 1, 1, name='score', reuse=reuse, c_i=self.num_units)
                 .log_softmax_high_dimension(self.num_classes, name='prob'))
            '''
            (self.feed('upscore', 'flow')
                 .rnn_gru2d_original(self.num_units, self.num_units, name='gru2d', reuse=reuse)
                 .conv(1, 1, self.num_classes, 1, 1, name='score', reuse=reuse, c_i=self.num_units)
                 .log_softmax_high_dimension(self.num_classes, name='prob'))
            '''

            (self.feed('score')
                 .softmax_high_dimension(self.num_classes, name='prob_normalized')
                 .argmax_2d(name='label_2d'))

            # collect outputs
            input_state = self.get_output('gru2d')[1]
            input_weights = self.get_output('gru2d')[2]
            input_points = self.get_output('flow')[2]
            outputs.append(self.get_output('prob'))
            probs.append(self.get_output('prob_normalized'))
            labels_gt_2d.append(self.get_output('gt_label_2d'))
            labels_pred_2d.append(self.get_output('label_2d'))

        self.layers['outputs'] = outputs
        self.layers['probs'] = probs
        self.layers['labels_gt_2d'] = labels_gt_2d
        self.layers['labels_pred_2d'] = labels_pred_2d
        self.layers['output_state'] = input_state
        self.layers['output_weights'] = input_weights
        self.layers['output_points'] = input_points
