import tensorflow as tf
import numpy as np

class GRU3DCell(tf.contrib.rnn.RNNCell):
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

    # inputs: [batch_size, grid_size, grid_size, grid_size, channels]
    # state:  [batch_size, grid_size, grid_size, grid_size, num_units]
    def __call__(self, inputs, flag, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            inputs_shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, [inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3], self._channels])

            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # concat inputs and state
                inputs_state = tf.concat(4, [inputs, state])

                # define the variables
                init_kernel = tf.constant_initializer(0.0)
                init_biases = tf.constant_initializer(0.0)
                kernel = self.make_var('weights', [1, 1, 1, self._num_units + self._channels, self._num_units], init_kernel)
                biases = self.make_var('biases', [self._num_units], init_biases)

                # 3D convolution
                conv = tf.nn.conv3d(inputs_state, kernel, [1, 1, 1, 1, 1], padding='SAME')
                u = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))

                # ru = tf.nn.sigmoid(ru)
                # r, u = tf.split(4, 2, ru)
            """
            with tf.variable_scope("Candidate"):
                inputs_rstate = tf.concat(4, [inputs, tf.mul(r, state)])

                # define the variables
                init_biases_1 = tf.constant_initializer(0.0)
                kernel_1 = self.make_var('weights', [1, 1, 1, self._num_units + self._channels, self._num_units])
                biases_1 = self.make_var('biases', [self._num_units], init_biases_1)

                # 3D convolution
                conv_1 = tf.nn.conv3d(inputs_rstate, kernel_1, [1, 1, 1, 1, 1], padding='SAME')
                # c = tf.nn.tanh(tf.nn.bias_add(conv_1, biases_1))
                c = tf.nn.bias_add(conv_1, biases_1)
            """
            new_state = tf.mul(flag, tf.nn.relu(u * state + (1 - u) * inputs))
            old_state = tf.mul(tf.sub(tf.ones(tf.shape(state), tf.float32), flag), state)
            new_h = new_state + old_state
        return new_h, new_h
