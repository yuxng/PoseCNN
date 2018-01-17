import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'hough_voting_gpu.so')
_hough_voting_gpu_module = tf.load_op_library(filename)
hough_voting_gpu = _hough_voting_gpu_module.houghvotinggpu
hough_voting_gpu_grad = _hough_voting_gpu_module.houghvotinggpu_grad
