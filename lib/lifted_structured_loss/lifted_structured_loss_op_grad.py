import tensorflow as tf
from tensorflow.python.framework import ops
import lifted_structured_loss_op

@ops.RegisterGradient("Liftedstruct")
def _liftedstruct_grad(op, grad, _):

  diff = op.outputs[1]
  margin = op.get_attr('margin')
  budget = op.get_attr('budget')

  # compute gradient
  data_grad = lifted_structured_loss_op.lifted_structured_loss_grad(diff, grad, margin, budget)

  return [data_grad, None]  # List of one Tensor, since we have two input
