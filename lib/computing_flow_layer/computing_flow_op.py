import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'computing_flow.so')
_computing_flow_module = tf.load_op_library(filename)
compute_flow = _computing_flow_module.computeflow
compute_flow_grad = _computing_flow_module.computeflow_grad
