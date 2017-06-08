import tensorflow as tf
import numpy as np
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'matching_loss.so')
_matching_loss_module = tf.load_op_library(filename)
matching_loss = _matching_loss_module.matching
matching_loss_grad = _matching_loss_module.matching_grad
