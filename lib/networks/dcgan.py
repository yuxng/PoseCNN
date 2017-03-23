import tensorflow as tf
from networks.network import Network

class dcgan(Network):
    def __init__(self, trainable=True):

        self.size = 128
        self.data = tf.placeholder(tf.float32, shape=[None, self.size, self.size, 3])
        self.data_gt = tf.placeholder(tf.float32, shape=[None, self.size, self.size, 3])
        self.z = tf.placeholder(tf.float32, shape=[None, 100])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.data, self.data_gt, self.z, self.keep_prob])
        data, data_gt, z, self.keep_prob_queue = q.dequeue()
        self.layers = dict({'data': data, 'data_gt': data_gt, 'z': z})
        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        self.trainable = trainable
        self.setup()

    def setup(self):
        # generator

        # project `z` and reshape
        (self.feed('z')
             .fc(int(self.size/32) * int(self.size/32) * 512, relu=False, name='fc_z', num_in=100)
             .reshape([-1, self.size/32, self.size/32, 512], name='reshape_z'))

        # encoder
        (self.feed('data')
             .conv(4, 4, 64, 2, 2, name='conv1', relu=False, c_i=3)
             .batch_norm(relu=True, name='bn1')
             .conv(4, 4, 128, 2, 2, name='conv2', relu=False, c_i=64)
             .batch_norm(relu=True, name='bn2')
             .conv(4, 4, 256, 2, 2, name='conv3', relu=False, c_i=128)
             .batch_norm(relu=True, name='bn3')
             .conv(4, 4, 512, 2, 2, name='conv4', relu=False, c_i=256)
             .batch_norm(relu=True, name='bn4')
             .conv(4, 4, 512, 2, 2, name='conv5', relu=False, c_i=512)
             .batch_norm(relu=True, name='bn5'))

        # decoder
        (self.feed('bn5', 'reshape_z')
             .concat(name='concat', axis=3)
             .deconv(4, 4, 512, 2, 2, name='deconv_1')
             .batch_norm(relu=True, name='bn1_deconv', c_i=512)
             .deconv(4, 4, 512, 2, 2, name='deconv_2')
             .batch_norm(relu=True, name='bn2_deconv', c_i=512)
             .deconv(4, 4, 256, 2, 2, name='deconv_3')
             .batch_norm(relu=True, name='bn3_deconv', c_i=256)
             .deconv(4, 4, 128, 2, 2, name='deconv_4')
             .batch_norm(relu=True, name='bn4_deconv', c_i=128)
             .deconv(4, 4, 64, 2, 2, name='deconv_5')
             .batch_norm(relu=True, name='bn5_deconv', c_i=64)
             .conv(1, 1, 3, 1, 1, name='conv_output', relu=False, c_i=64)
             .tanh(name='output_g'))

        # discriminator
        outputs_d = []
        for i in range(2):
            print i
            if i == 0:
                reuse = None
                self.layers['input_d'] = self.layers['output_g']
            else:
                reuse = True
                self.layers['input_d'] = self.layers['data_gt']

            (self.feed('input_d', 'data')
                 .concat(3, name='image_d')
                 .conv(4, 4, 64, 2, 2, name='conv1_d', relu=False, reuse=reuse, c_i=6)
                 .lrelu(name='lrelu1_d')
                 .conv(4, 4, 128, 2, 2, name='conv2_d', relu=False, reuse=reuse, c_i=64)
                 .batch_norm(relu=False, name='bn2_d', reuse=reuse)
                 .lrelu(name='lrelu2_d')
                 .conv(4, 4, 256, 2, 2, name='conv3_d', relu=False, reuse=reuse, c_i=128)
                 .batch_norm(relu=False, name='bn3_d', reuse=reuse)
                 .lrelu(name='lrelu3_d')
                 .conv(4, 4, 512, 2, 2, name='conv4_d', relu=False, reuse=reuse, c_i=256)
                 .batch_norm(relu=False, name='bn4_d', reuse=reuse)
                 .lrelu(name='lrelu4_d')
                 .conv(4, 4, 512, 2, 2, name='conv5_d', relu=False, reuse=reuse, c_i=512)
                 .batch_norm(relu=False, name='bn5_d', reuse=reuse)
                 .lrelu(name='lrelu5_d')
                 .reshape([-1, int(self.size/32) * int(self.size/32) * 512], name='reshape_d')
                 .fc(1, relu=False, name='fc_d', reuse=reuse)
                 .sigmoid(name='sigmoid_d'))

            # collect outputs
            outputs_d.append(self.get_output('fc_d'))

        self.layers['outputs_d'] = outputs_d
