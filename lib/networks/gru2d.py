import tensorflow as tf

class GRU2DCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

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
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # concat inputs and state
                inputs_state = tf.concat(3, [inputs, state])

                # define the variables
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(1.0)
                kernel = self.make_var('weights', [1, 1, self._num_units + self._channels, 2 * self._num_units], init_weights)
                biases = self.make_var('biases', [2 * self._num_units], init_biases)

                # 2D convolution
                conv = tf.nn.conv2d(inputs_state, kernel, [1, 1, 1, 1], padding='SAME')
                ru = tf.nn.bias_add(conv, biases)

                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(3, 2, ru)

            with tf.variable_scope("Candidate"):
                inputs_rstate = tf.concat(3, [inputs, tf.mul(r, state)])

                # define the variables
                init_biases_1 = tf.constant_initializer(0.0)
                kernel_1 = self.make_var('weights', [1, 1, self._num_units + self._channels, self._num_units], init_weights)
                biases_1 = self.make_var('biases', [self._num_units], init_biases_1)

                # 2D convolution
                conv_1 = tf.nn.conv2d(inputs_rstate, kernel_1, [1, 1, 1, 1], padding='SAME')
                c = tf.nn.tanh(tf.nn.bias_add(conv_1, biases_1))

            new_h = u * state + (1 - u) * c
        return new_h, new_h
