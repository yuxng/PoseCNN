import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'hard_label.so')
_hard_label_module = tf.load_op_library(filename)
hard_label = _hard_label_module.hardlabel
hard_label_grad = _hard_label_module.hardlabel_grad
