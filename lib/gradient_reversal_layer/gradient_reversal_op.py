import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'gradient_reversal.so')
_gradient_reversal_module = tf.load_op_library(filename)
gradient_reversal = _gradient_reversal_module.gradientreversal
gradient_reversal_grad = _gradient_reversal_module.gradientreversal_grad
