import numpy as np
from math import ceil
import tensorflow as tf
import backprojecting_layer.backprojecting_op as backproject_op
import backprojecting_layer.backprojecting_op_grad
import projecting_layer.projecting_op as project_op
import projecting_layer.projecting_op_grad
import computing_label_layer.computing_label_op as compute_label_op
from gru2d import GRU2DCell
from vanilla2d import Vanilla2DCell
from add2d import Add2DCell

DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            print op_name
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def make_deconv_filter(self, name, f_shape, trainable=True):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)
        return var

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        if isinstance(input, tuple):
            input = input[0]
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)
            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)    
        return tf.nn.bias_add(conv, biases, name=scope.name)


    @layer
    def conv3d(self, input, k_d, k_h, k_w, c_i, c_o, s_d, s_h, s_w, name, reuse=None, relu=True, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        if isinstance(input, tuple):
            input = input[0]
        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_d, k_h, k_w, c_i, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)
            conv = tf.nn.conv3d(input, kernel, [1, s_d, s_h, s_w, 1], padding=padding)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def deconv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        with tf.variable_scope(name, reuse=reuse) as scope:
            # Compute shape out of input
            in_shape = tf.shape(input)
            h = in_shape[1] * s_h
            w = in_shape[2] * s_w
            new_shape = [in_shape[0], h, w, c_o]
            output_shape = tf.pack(new_shape)

            # filter
            f_shape = [k_h, k_w, c_o, c_i]
            weights = self.make_deconv_filter('weights', f_shape, trainable)
        return tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)

    @layer
    def backproject(self, input, grid_size, threshold, name):
        return backproject_op.backproject(input[0][1], input[1], input[2], input[3], grid_size, threshold, name=name)

    @layer
    def project(self, input, threshold, name):
        return project_op.project(input[0], input[1], input[2], threshold, name=name)

    @layer
    def compute_label(self, input, name):
        return compute_label_op.compute_label(input[0], input[1], input[2], name=name)

    @layer
    def rnn_gru2d(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            gru2d = GRU2DCell(num_units, channels)
            return gru2d(input[0], input[1], scope)

    @layer
    def rnn_vanilla2d(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            vanilla2d = Vanilla2DCell(num_units, channels)
            return vanilla2d(input[0], input[1], scope)
    
    @layer
    def rnn_add2d(self, input, num_units, channels, step, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            add2d = Add2DCell(num_units, channels)
            return add2d(input[0], input[1], step, scope)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add(inputs[0], inputs[1])

    @layer
    def fc(self, input, num_out, name, reuse=None, relu=True, trainable=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.softmax(input, name)

    @layer
    def softmax_high_dimension(self, input, num_classes, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        print input
        input_shape = input.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes

        m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        e = tf.exp(tf.sub(input, tf.tile(m, multiples)))
        s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
        return tf.div(e, tf.tile(s, multiples))


    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
