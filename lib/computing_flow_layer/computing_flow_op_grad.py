import tensorflow as tf
from tensorflow.python.framework import ops
import computing_flow_op
'''
@tf.RegisterShape("Computeflow")
def _computeflow_shape(op):
  """Shape function for the Computeflow op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  height = dims_data[1]
  width = dims_data[2]
  channels = dims_data[3]

  output_shape = tf.TensorShape([batch_size, height, width, channels])
  output_shape_points = tf.TensorShape([batch_size, height, width, 3])
  return [output_shape, output_shape_points]
'''
@ops.RegisterGradient("Computeflow")
def _computeflow_grad(op, grad, _):
  """The gradients for `computeflow`.
  Args:
    op: The `computeflow` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `computeflow` op.
  Returns:
    Gradients with respect to the input of `computeflow`.
  """
  data = op.inputs[0]
  bottom_points = op.inputs[1]
  top_points = op.outputs[1]
  kernel_size = op.get_attr('kernel_size')
  threshold = op.get_attr('threshold')

  # compute gradient
  data_grad = computing_flow_op.compute_flow_grad(data, bottom_points, top_points, grad, kernel_size, threshold)

  return [data_grad, None, None, None]  # List of one Tensor, since we have four input
