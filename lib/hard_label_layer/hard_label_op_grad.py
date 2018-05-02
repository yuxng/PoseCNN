import tensorflow as tf
from tensorflow.python.framework import ops
import hard_label_op

@ops.RegisterShape("Hardlabel")
def _hard_label_shape(op):

  output_shape = op.inputs[0].get_shape()
  return [output_shape]

@ops.RegisterGradient("Hardlabel")
def _hard_label_grad(op, grad):

  bottom_prob = op.inputs[0]
  bottom_gt = op.inputs[1]
  threshold = op.get_attr('threshold')

  # compute gradient
  data_grad_prob, data_grad_gt = hard_label_op.hard_label_grad(bottom_prob, bottom_gt, grad, threshold)

  return [data_grad_prob, data_grad_gt]
