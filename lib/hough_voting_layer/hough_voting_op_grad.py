import tensorflow as tf
from tensorflow.python.framework import ops
import hough_voting_op

@ops.RegisterShape("Houghvoting")
def _hough_voting_shape(op):

  output_shape = tf.TensorShape([None, 6])
  return [output_shape]

@ops.RegisterGradient("Houghvoting")
def _hough_voting_grad(op, grad):
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

  return [data_grad_prob, data_grad_vertex]  # List of one Tensor, since we have two input
