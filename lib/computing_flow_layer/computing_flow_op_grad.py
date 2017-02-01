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
def _computeflow_grad(op, grad, grad_weights, _):
  """The gradients for `computeflow`.
  Args:
    op: The `computeflow` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `computeflow` op.
  Returns:
    Gradients with respect to the input of `computeflow`.
  """
  data = op.inputs[0]
  bottom_weights = op.inputs[1]
  bottom_points = op.inputs[2]
  bottom_depth = op.inputs[3]
  bottom_meta_data = op.inputs[4]
  top_points = op.outputs[2]
  kernel_size = op.get_attr('kernel_size')
  threshold = op.get_attr('threshold')
  max_weight = op.get_attr('max_weight')

  # compute gradient
  data_grad, data_grad_weights = computing_flow_op.compute_flow_grad(data, bottom_weights, bottom_points, bottom_depth, bottom_meta_data, \
      top_points, grad, grad_weights, kernel_size, threshold, max_weight)

  return [data_grad, data_grad_weights, None, None, None]  # List of one Tensor, since we have five input
