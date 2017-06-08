import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'hough_voting.so')
_hough_voting_module = tf.load_op_library(filename)
hough_voting = _hough_voting_module.houghvoting
hough_voting_grad = _hough_voting_module.houghvoting_grad
