import tensorflow as tf
from tensorflow.python.framework import ops
import backprojecting_op

@tf.RegisterShape("Backproject")
def _backproject_shape(op):
  """Shape function for the Backproject op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  height = dims_data[1]
  width = dims_data[2]
  channels = dims_data[3]

  dims_pixel_locations = op.inputs[1].get_shape().as_list()
  grid_size = dims_pixel_locations[1]

  output_shape_1 = tf.TensorShape([batch_size, grid_size, grid_size, grid_size, channels])
  output_shape_2 = tf.TensorShape([batch_size, grid_size, grid_size, grid_size, 1])
  output_shape_3 = tf.TensorShape([batch_size, height, width, 1])
  return [output_shape_1, output_shape_2, output_shape_3]

@ops.RegisterGradient("Backproject")
def _backproject_grad(op, grad, tmp, _):
  """The gradients for `backproject`.
  Args:
    op: The `backproject` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `backproject` op.
  Returns:
    Gradients with respect to the input of `backproject`.
  """
  data = op.inputs[0]
  top_count = op.outputs[1]
  top_voxel_locations = op.outputs[2]

  # compute gradient
  data_grad = backprojecting_op.backproject_grad(data, top_count, top_voxel_locations, grad)

  return [data_grad, None]  # List of one Tensor, since we have one input
