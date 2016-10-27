import tensorflow as tf
from tensorflow.python.framework import ops
import backprojecting_op

@tf.RegisterShape("Backproject")
def _backproject_shape(op):
  """Shape function for the Backproject op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  channels = dims_data[3]
  grid_size = op.get_attr('grid_size')

  output_shape = tf.TensorShape([batch_size, grid_size, grid_size, grid_size, channels])
  return [output_shape]

@ops.RegisterGradient("Backproject")
def _backproject_grad(op, grad):
  """The gradients for `backproject`.
  Args:
    op: The `backproject` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `backproject` op.
  Returns:
    Gradients with respect to the input of `backproject`.
  """
  data = op.inputs[0]
  depth = op.inputs[2]
  meta_data = op.inputs[3]
  grid_size = op.get_attr('grid_size')
  threshold = op.get_attr('threshold')

  # compute gradient
  data_grad = backprojecting_op.backproject_grad(data, depth, meta_data, grad, grid_size, threshold)

  return [data_grad, None, None, None]  # List of one Tensor, since we have four input
