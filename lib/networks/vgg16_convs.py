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
            self.input_dim = 6
        else:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.conv1_name = 'conv1_1'
            self.input_dim = 3

        self.gt_label_2d = tf.placeholder(tf.float32, shape=[None, None, None, self.num_classes])
        self.depth = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[None, None, None, 48])

        # define a queue
        q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.depth, self.meta_data])
        data, gt_label_2d, depth, meta_data = q.dequeue()

        self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d, 'depth': depth, 'meta_data': meta_data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name=self.conv1_name, c_i=self.input_dim)
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
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512)
             .conv(1, 1, 64, 1, 1, name='score_conv5', c_i=512)
             .deconv(4, 4, 64, 2, 2, name='upscore_conv5', trainable=False))

        (self.feed('conv4_3')
             .conv(1, 1, 64, 1, 1, name='score_conv4', c_i=512))

        (self.feed('score_conv4', 'upscore_conv5')
             .add(name='add1')
             .deconv(int(16*self.scale), int(16*self.scale), 64, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False)
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=64)
             .meanfield_2d(3, self.num_classes, name='meanfield')
             .log_softmax_high_dimension(self.num_classes, name='prob')
             .argmax_2d(name='label_2d'))
