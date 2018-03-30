import tensorflow as tf
from tensorflow.python.framework import ops
import gradient_reversal_op

@ops.RegisterShape("Gradientreversal")
def _gradient_reversal_shape(op):

  output_shape = op.inputs[0].get_shape()
  return [output_shape]

@ops.RegisterGradient("Gradientreversal")
def _gradient_reversal_grad(op, grad):

  bottom_data = op.inputs[0]
  lambda_ = op.get_attr('lambda')

  # compute gradient
  data_grad = gradient_reversal_op.gradient_reversal_grad(bottom_data, grad, lambda_)

  return [data_grad]
