import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'triplet_loss.so')
_triplet_loss_module = tf.load_op_library(filename)
triplet_loss = _triplet_loss_module.triplet
triplet_loss_grad = _triplet_loss_module.triplet_grad
