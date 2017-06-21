import tensorflow as tf

class Add2DCell(tf.contrib.rnn.RNNCell):
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

    # inputs: [batch_size, height, width, channels]
    # state:  [batch_size, height, width, num_units]
    def __call__(self, inputs, state, step, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "Add2DCell"
            new_h = (inputs + step * state) / (step + 1)
        return new_h, new_h
