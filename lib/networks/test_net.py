import tensorflow as tf
from networks.network import Network

class test_net(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.depth = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.label = tf.placeholder(tf.int32, shape=[None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[None, None, None, 33])
        self.layers = dict({'data':self.data, 'depth':self.depth, 'label':self.label, 'meta_data':self.meta_data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 7, 1, 1, name='conv'))

        (self.feed('conv', 'depth', 'label', 'meta_data')
             .backproject(256, 7, 0.001, name='backprojection')
             .softmax_high_dimension(num_classes=7, name='prob'))
