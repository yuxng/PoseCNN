import tensorflow as tf
from networks.network import Network

class vgg16_convs(Network):
    def __init__(self, grid_size, trainable=True):
        self.inputs = []
        self.num_classes = 7
        self.grid_size = grid_size

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.label = tf.placeholder(tf.float32, shape=[None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[None, None, None, 33])

        self.layers = dict({'data': self.data, 'label': self.label, 'depth': self.depth, 'meta_data': self.meta_data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .conv(1, 1, self.num_classes, 1, 1, name='score')
             .deconv(32, 32, self.num_classes, 16, 16, name='score_up'))

        (self.feed('score_up', 'label', 'depth', 'meta_data')
             .backproject(self.grid_size, 0.02, name='backprojection')
             .log_softmax_high_dimension(self.num_classes, name='prob'))

        (self.feed('prob', 'depth', 'meta_data')
             .compute_label(name='label'))
