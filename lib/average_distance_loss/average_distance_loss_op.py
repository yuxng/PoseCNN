import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'average_distance_loss.so')
_average_distance_loss_module = tf.load_op_library(filename)
average_distance_loss = _average_distance_loss_module.averagedistance
average_distance_loss_grad = _average_distance_loss_module.averagedistance_grad
