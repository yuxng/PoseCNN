import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'backprojecting.so')
_backprojecting_module = tf.load_op_library(filename)
backproject = _backprojecting_module.backproject
backproject_grad = _backprojecting_module.backproject_grad
