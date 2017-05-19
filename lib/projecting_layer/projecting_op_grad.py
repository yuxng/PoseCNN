import tensorflow as tf
from tensorflow.python.framework import ops
import projecting_op
'''
@tf.RegisterShape("Project")
def _project_shape(op):
  """Shape function for the Backproject op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  channels = dims_data[4]

  dims_image = op.inputs[1].get_shape().as_list()
  height = dims_image[1]
  width = dims_image[2]

  output_shape = tf.TensorShape([batch_size, height, width, channels])
  return [output_shape]
'''
@ops.RegisterGradient("Project")
def _project_grad(op, grad):
  """The gradients for `project`.
  Args:
    op: The `project` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `backproject` op.
  Returns:
    Gradients with respect to the input of `project`.
  """

  data = op.inputs[0]
  depth = op.inputs[1]
  meta_data = op.inputs[2]
  kernel_size = op.get_attr('kernel_size')
  threshold = op.get_attr('threshold')

  # compute gradient
  data_grad = projecting_op.project_grad(data, depth, meta_data, grad, kernel_size, threshold)

  return [data_grad, None, None]  # List of one Tensor, since we have three input
