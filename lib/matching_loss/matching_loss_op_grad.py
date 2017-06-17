import tensorflow as tf
from tensorflow.python.framework import ops
import matching_loss_op

@ops.RegisterShape("Matching")
def _matching_shape(op):
  """Shape function for the Matching op.

  """

  output_shape = tf.TensorShape([1])
  output_shape_1 = op.inputs[0].get_shape()
  return [output_shape, output_shape_1]

@ops.RegisterGradient("Matching")
def _matching_grad(op, grad, _):

  diff = op.outputs[1]

  # compute gradient
  data_grad = matching_loss_op.matching_loss_grad(diff, grad)

  return [data_grad, None, None, None, None, None, None]  # List of one Tensor, since we have three input
