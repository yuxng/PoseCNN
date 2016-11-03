import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'projecting.so')
_projecting_module = tf.load_op_library(filename)
project = _projecting_module.project
project_grad = _projecting_module.project_grad
