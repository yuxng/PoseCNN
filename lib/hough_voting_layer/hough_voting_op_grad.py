import tensorflow as tf
from tensorflow.python.framework import ops
import hough_voting_op

@ops.RegisterShape("Houghvoting")
def _hough_voting_shape(op):

  dims_vertex = op.inputs[1].get_shape().as_list()
  num_classes = dims_vertex[3] / 3

  output_shape_0 = tf.TensorShape([None, 6])
  output_shape_1 = tf.TensorShape([None, 7])
  output_shape_2 = tf.TensorShape([None, 4 * num_classes])
  output_shape_3 = tf.TensorShape([None, 4 * num_classes])
  return [output_shape_0, output_shape_1, output_shape_2, output_shape_3]

@ops.RegisterGradient("Houghvoting")
def _hough_voting_grad(op, grad, tmp, tmp1, _):
  """The gradients for `Houghvoting`.
  Args:
    op: The `backproject` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `backproject` op.
  Returns:
    Gradients with respect to the input of `backproject`.
  """

  bottom_prob = op.inputs[0]
  bottom_vertex = op.inputs[1]

  # compute gradient
  data_grad_prob, data_grad_vertex = hough_voting_op.hough_voting_grad(bottom_prob, bottom_vertex, grad)

  return [data_grad_prob, data_grad_vertex, None, None, None]  # List of one Tensor, since we have two input
