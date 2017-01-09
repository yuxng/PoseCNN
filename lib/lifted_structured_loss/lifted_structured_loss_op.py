import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'lifted_structured_loss.so')
_lifted_structured_loss_module = tf.load_op_library(filename)
lifted_structured_loss = _lifted_structured_loss_module.liftedstruct
lifted_structured_loss_grad = _lifted_structured_loss_module.liftedstruct_grad
