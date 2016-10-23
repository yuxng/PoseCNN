import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'computing_label.so')
_computing_label_module = tf.load_op_library(filename)
compute_label = _computing_label_module.computelabel
