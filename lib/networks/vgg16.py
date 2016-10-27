import tensorflow as tf
from networks.network import Network

class vgg16(Network):
    def __init__(self, num_steps, trainable=True):
        self.inputs = []
        self.grid_size = 256
        self.num_classes = 7
        self.num_steps = num_steps

        self.data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 3])
        self.depth = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 1])
        self.meta_data = tf.placeholder(tf.float32, shape=[self.num_steps, None, None, None, 33])
        self.state = tf.placeholder(tf.float32, [None, self.grid_size, self.grid_size, self.grid_size, 8])
        self.layers = dict({'data': [], 'depth': [], 'meta_data': [], 'state_3d': []})

        self.trainable = trainable
        self.setup()

    def setup(self):
        input_data = tf.unpack(self.data)
        input_depth = tf.unpack(self.depth)
        input_meta_data = tf.unpack(self.meta_data)
        input_state = self.state
        output = []
        
        for i in range(self.num_steps):
            # set inputs
            self.layers['data'] = input_data[i]
            self.layers['depth'] = input_depth[i]
            self.layers['meta_data'] = input_meta_data[i]
            self.layers['state_3d'] = input_state
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
                 .conv(3, 3, 64, 1, 1, name='conv6', reuse=reuse)
                 .deconv(32, 32, 64, 16, 16, name='conv6_up', reuse=reuse))

            (self.feed('state_3d', 'depth', 'meta_data')
                 .project(name='state_2d'))

            (self.feed('conv6_up', 'state_2d')
                 .rnn_gru2d(8, 64, name='gru2d', reuse=reuse)
                 .conv(1, 1, self.num_classes, 1, 1, name='score', reuse=reuse))

            (self.feed('gru2d', 'state_3d', 'depth', 'meta_data')
                 .backproject(self.grid_size, 0.01, name='backprojection'))

            # collect outputs
            input_state = self.get_output('backprojection')
            output.append(self.get_output('score'))

        self.layers['output'] = output
