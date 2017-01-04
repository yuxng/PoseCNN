import tensorflow as tf
from networks.network import Network

class vgg16(Network):
    def __init__(self, input_format, num_steps, num_classes, num_units, scales, trainable=True):
        self.inputs = []
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_units = num_units
        self.scale = 1 / scales[0]

        if input_format == 'RGBD':
            self.data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 6])
            self.conv1_name = 'conv1_1_new'
        else:
            self.data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 3])
            self.conv1_name = 'conv1_1'

        self.gt_label_2d = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 48])
        self.state = tf.placeholder(tf.float32, [None, None, None, self.num_units])
        self.points = tf.placeholder(tf.float32, [None, None, None, 3])
        self.layers = dict({'data': [], 'gt_label_2d': [], 'depth': [], 'meta_data': [], 'state': [], 'points': []})
        self.trainable = trainable
        self.setup()

    def setup(self):
        input_data = tf.unpack(self.data)
        input_label_2d = tf.unpack(self.gt_label_2d)
        input_depth = tf.unpack(self.depth)
        input_meta_data = tf.unpack(self.meta_data)
        input_state = self.state
        input_points = self.points
        outputs = []
        labels_gt_2d = []
        labels_pred_2d = []
        
        for i in range(self.num_steps):
            # set inputs
            self.layers['data'] = input_data[i]
            self.layers['gt_label_2d'] = input_label_2d[i]
            self.layers['depth'] = input_depth[i]
            self.layers['meta_data'] = input_meta_data[i]
            self.layers['state'] = input_state
            self.layers['points'] = input_points
            if i == 0:
                reuse = None
            else:
                reuse = True

            (self.feed('data')
                 .conv(3, 3, 64, 1, 1, name=self.conv1_name, reuse=reuse)
                 .conv(3, 3, 64, 1, 1, name='conv1_2', reuse=reuse)
                 .max_pool(2, 2, 2, 2, name='pool1')
                 .conv(3, 3, 128, 1, 1, name='conv2_1', reuse=reuse)
                 .conv(3, 3, 128, 1, 1, name='conv2_2', reuse=reuse)
                 .max_pool(2, 2, 2, 2, name='pool2')
                 .conv(3, 3, 256, 1, 1, name='conv3_1', reuse=reuse)
                 .conv(3, 3, 256, 1, 1, name='conv3_2', reuse=reuse)
                 .conv(3, 3, 256, 1, 1, name='conv3_3', reuse=reuse)
                 .max_pool(2, 2, 2, 2, name='pool3')
                 .conv(3, 3, 512, 1, 1, name='conv4_1', reuse=reuse)
                 .conv(3, 3, 512, 1, 1, name='conv4_2', reuse=reuse)
                 .conv(3, 3, 512, 1, 1, name='conv4_3', reuse=reuse)
                 .max_pool(2, 2, 2, 2, name='pool4')
                 .conv(3, 3, 512, 1, 1, name='conv5_1', reuse=reuse)
                 .conv(3, 3, 512, 1, 1, name='conv5_2', reuse=reuse)
                 .conv(3, 3, 512, 1, 1, name='conv5_3', reuse=reuse)
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv5', reuse=reuse)
                 .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', reuse=reuse, trainable=False))

            (self.feed('conv4_3')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv4', reuse=reuse))

            (self.feed('score_conv4', 'upscore_conv5')
                 .add(name='add1')
                 .deconv(int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', reuse=reuse, trainable=False))

            (self.feed('state', 'points', 'depth', 'meta_data')
                 .compute_flow(5, 0.02, name='flow'))

            (self.feed('upscore', 'flow')
                 .rnn_gru2d(self.num_units, self.num_units, name='gru2d', reuse=reuse)
                 .conv(1, 1, self.num_classes, 1, 1, name='score', reuse=reuse)
                 .log_softmax_high_dimension(self.num_classes, name='prob')
                 .argmax_2d(name='label_2d'))

            # collect outputs
            input_state = self.get_output('gru2d')[1]
            input_points = self.get_output('flow')[1]
            outputs.append(self.get_output('prob'))
            labels_gt_2d.append(self.get_output('gt_label_2d'))
            labels_pred_2d.append(self.get_output('label_2d'))

        self.layers['outputs'] = outputs
        self.layers['labels_gt_2d'] = labels_gt_2d
        self.layers['labels_pred_2d'] = labels_pred_2d
        self.layers['output_state'] = input_state
        self.layers['output_points'] = input_points
