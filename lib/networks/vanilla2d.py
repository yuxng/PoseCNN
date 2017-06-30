import tensorflow as tf

class Vanilla2DCell(tf.contrib.rnn.RNNCell):
    """Vanilla Recurrent Unit cell."""

    def __init__(self, num_units, channels):
        self._num_units = num_units
        self._channels = channels

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    # inputs: [batch_size, height, width, channels]
    # state:  [batch_size, height, width, num_units]
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "VanillaCell"
            inputs_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [inputs_shape[0], inputs_shape[1], inputs_shape[2], self._channels])

            # concat inputs and state
            inputs_state = tf.concat(3, [inputs, state])

            # define the variables
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [3, 3, self._num_units + self._channels, self._num_units])
            biases = self.make_var('biases', [self._num_units], init_biases)

            # 2D convolution
            conv = tf.nn.conv2d(inputs_state, kernel, [1, 1, 1, 1], padding='SAME')
            new_h = tf.nn.tanh(tf.nn.bias_add(conv, biases))

        return new_h, new_h
