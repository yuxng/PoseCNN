import tensorflow as tf
from networks.network import Network

class vgg16_convs(Network):
    def __init__(self, input_format, num_classes, scales, trainable=True):
        self.inputs = []
        self.num_classes = num_classes
        self.scale = 1 / scales[0]

        if input_format == 'RGBD':
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 6])
            self.conv1_name = 'conv1_1_new'
        else:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.conv1_name = 'conv1_1'

        self.gt_label_2d = tf.placeholder(tf.float32, shape=[None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[None, None, None, 48])

        self.layers = dict({'data': self.data, 'gt_label_2d': self.gt_label_2d, 'depth': self.depth, 'meta_data': self.meta_data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name=self.conv1_name)
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
             .conv(1, 1, 64, 1, 1, name='score_conv5')
             .deconv(4, 4, 64, 2, 2, name='upscore_conv5', trainable=False))

        (self.feed('conv4_3')
             .conv(1, 1, 64, 1, 1, name='score_conv4'))

        (self.feed('score_conv4', 'upscore_conv5')
             .add(name='add1')
             .deconv(int(16*self.scale), int(16*self.scale), 64, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False)
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=64)
             .log_softmax_high_dimension(self.num_classes, name='prob')
             .argmax_2d(name='label_2d'))
