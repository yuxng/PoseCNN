import tensorflow as tf
from networks.network import Network

class resnet50(Network):
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
        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        data, gt_label_2d, depth, meta_data = q.dequeue()

        self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d, 'depth': depth, 'meta_data': meta_data})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, relu=False, name='conv1', c_i=self.input_dim)
             .batch_normalization(relu=True, name='bn_conv1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1', c_i=64)
             .batch_normalization(name='bn2a_branch1'))

        (self.feed('bn_conv1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a', c_i=64)
             .batch_normalization(relu=True, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b', c_i=64)
             .batch_normalization(relu=True, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c', c_i=64)
             .batch_normalization(name='bn2a_branch2c'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a', c_i=256)
             .batch_normalization(relu=True, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b', c_i=64)
             .batch_normalization(relu=True, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c', c_i=64)
             .batch_normalization(name='bn2b_branch2c'))

        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a', c_i=256)
             .batch_normalization(relu=True, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b', c_i=64)
             .batch_normalization(relu=True, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c', c_i=64)
             .batch_normalization(name='bn2c_branch2c'))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1', c_i=256)
             .batch_normalization(name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a', c_i=256)
             .batch_normalization(relu=True, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b', c_i=128)
             .batch_normalization(relu=True, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c', c_i=128)
             .batch_normalization(name='bn3a_branch2c'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a', c_i=512)
             .batch_normalization(relu=True, name='bn3b_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b', c_i=128)
             .batch_normalization(relu=True, name='bn3b_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c', c_i=128)
             .batch_normalization(name='bn3b_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b_branch2c')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a', c_i=512)
             .batch_normalization(relu=True, name='bn3c_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b', c_i=128)
             .batch_normalization(relu=True, name='bn3c_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c', c_i=128)
             .batch_normalization(name='bn3c_branch2c'))

        (self.feed('res3b_relu', 
                   'bn3c_branch2c')
             .add(name='res3c')
             .relu(name='res3c_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a', c_i=512)
             .batch_normalization(relu=True, name='bn3d_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b', c_i=128)
             .batch_normalization(relu=True, name='bn3d_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c', c_i=128)
             .batch_normalization(name='bn3d_branch2c'))

        (self.feed('res3c_relu', 
                   'bn3d_branch2c')
             .add(name='res3d')
             .relu(name='res3d_relu')
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1', c_i=512)
             .batch_normalization(name='bn4a_branch1'))

        (self.feed('res3d_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a', c_i=512)
             .batch_normalization(relu=True, name='bn4a_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c', c_i=256)
             .batch_normalization(name='bn4a_branch2c'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn4b_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4b_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c', c_i=256)
             .batch_normalization(name='bn4b_branch2c'))

        (self.feed('res4a_relu', 
                   'bn4b_branch2c')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn4c_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4c_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4c_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c', c_i=256)
             .batch_normalization(name='bn4c_branch2c'))

        (self.feed('res4b_relu', 
                   'bn4c_branch2c')
             .add(name='res4c')
             .relu(name='res4c_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn4d_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4d_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4d_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c', c_i=256)
             .batch_normalization(name='bn4d_branch2c'))

        (self.feed('res4c_relu', 
                   'bn4d_branch2c')
             .add(name='res4d')
             .relu(name='res4d_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn4e_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4e_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4e_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c', c_i=256)
             .batch_normalization(name='bn4e_branch2c'))

        (self.feed('res4d_relu', 
                   'bn4e_branch2c')
             .add(name='res4e')
             .relu(name='res4e_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn4f_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4f_branch2b', c_i=256)
             .batch_normalization(relu=True, name='bn4f_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c', c_i=256)
             .batch_normalization(name='bn4f_branch2c'))

        (self.feed('res4e_relu', 
                   'bn4f_branch2c')
             .add(name='res4f')
             .relu(name='res4f_relu')
             .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1', c_i=1024)
             .batch_normalization(name='bn5a_branch1'))

        (self.feed('res4f_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a', c_i=1024)
             .batch_normalization(relu=True, name='bn5a_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b', c_i=512)
             .batch_normalization(relu=True, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c', c_i=512)
             .batch_normalization(name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a', c_i=2048)
             .batch_normalization(relu=True, name='bn5b_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b', c_i=512)
             .batch_normalization(relu=True, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c', c_i=512)
             .batch_normalization(name='bn5b_branch2c'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a', c_i=2048)
             .batch_normalization(relu=True, name='bn5c_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b', c_i=512)
             .batch_normalization(relu=True, name='bn5c_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c', c_i=512)
             .batch_normalization(name='bn5c_branch2c'))

        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=2048)
             .deconv(int(32*self.scale), int(32*self.scale), self.num_classes, int(16*self.scale), int(16*self.scale), name='upscore', trainable=False)
             .log_softmax_high_dimension(self.num_classes, name='prob')
             .argmax_2d(name='label_2d'))
