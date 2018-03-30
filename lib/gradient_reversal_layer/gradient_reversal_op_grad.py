import tensorflow as tf
from tensorflow.python.framework import ops
import gradient_reversal_op

@ops.RegisterShape("Gradientreversal")
def _gradient_reversal_shape(op):

  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  height = dims_data[1]
  width = dims_data[2]
  channels = dims_data[3]

  output_shape = tf.TensorShape([batch_size, height, width, channels])
  return [output_shape]

@ops.RegisterGradient("Gradientreversal")
def _gradient_reversal_grad(op, grad):

  bottom_data = op.inputs[0]
  lambda_ = op.get_attr('lambda')

  # compute gradient
  data_grad = gradient_reversal_op.gradient_reversal_grad(bottom_data, grad, lambda_)

  return [data_grad]
