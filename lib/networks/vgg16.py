import tensorflow as tf
from networks.network import Network

class vgg16(Network):
    def __init__(self, grid_size, num_steps, num_units, trainable=True):
        self.inputs = []
        self.num_classes = 7
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.num_units = num_units

        self.data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 3])
        self.label = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 48])
        self.state = tf.placeholder(tf.float32, [None, self.grid_size, self.grid_size, self.grid_size, self.num_units])
        self.label_3d = tf.placeholder(tf.float32, [None, self.grid_size, self.grid_size, self.grid_size, self.num_classes])
        self.layers = dict({'data': [], 'label': [], 'depth': [], 'meta_data': [], 'state': [], 'label_3d': []})
        self.trainable = trainable
        self.setup()

    def setup(self):
        input_data = tf.unpack(self.data)
        input_label = tf.unpack(self.label)
        input_depth = tf.unpack(self.depth)
        input_meta_data = tf.unpack(self.meta_data)
        input_state = self.state
        input_label_3d = self.label_3d
        outputs = []
        labels_gt = []
        labels_pred = []
        
        for i in range(self.num_steps):
            # set inputs
            self.layers['data'] = input_data[i]
            self.layers['label'] = input_label[i]
            self.layers['depth'] = input_depth[i]
            self.layers['meta_data'] = input_meta_data[i]
            self.layers['state'] = input_state
            self.layers['label_3d'] = input_label_3d
            if i == 0:
                reuse = None
            else:
                reuse = True

            (self.feed('data')
                 .conv(3, 3, 64, 1, 1, name='conv1_1', reuse=reuse)
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
                 .conv(1, 1, self.num_classes, 1, 1, name='score_conv5', reuse=reuse)
                 .deconv(4, 4, self.num_classes, 2, 2, name='upscore_conv5', reuse=reuse, trainable=False))

            (self.feed('conv4_3')
                 .conv(1, 1, self.num_classes, 1, 1, name='score_conv4', reuse=reuse))

            (self.feed('score_conv4', 'upscore_conv5')
                 .add(name='add1')
                 .deconv(16, 16, self.num_classes, 8, 8, name='upscore', reuse=reuse, trainable=False))

            (self.feed('upscore', 'label', 'depth', 'meta_data', 'label_3d')
                 .backproject(self.grid_size, 0.02, name='backprojection'))

            (self.feed('backprojection', 'state')
                 .rnn_gru3d(self.num_units, self.num_classes, name='gru3d', reuse=reuse)
                 .meanfield_3d(self.num_classes, name='meanfield_3d', reuse=reuse)
                 .log_softmax_high_dimension(self.num_classes, name='prob'))

            (self.feed('prob', 'depth', 'meta_data')
                 .compute_label(name='label'))

            # collect outputs
            input_state = self.get_output('gru3d')[1]
            input_label_3d = self.get_output('backprojection')[1]
            outputs.append(self.get_output('prob'))
            labels_gt.append(self.get_output('backprojection')[1])
            labels_pred.append(self.get_output('label'))

        self.layers['outputs'] = outputs
        self.layers['labels_gt'] = labels_gt
        self.layers['labels_pred'] = labels_pred
        self.layers['output_state'] = input_state
        self.layers['output_label_3d'] = input_label_3d
