import tensorflow as tf
import numpy as np
import backprojecting_op
import backprojecting_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

batch_size = 32
height = 100
width = 100
grid_size = 10
channels = 4
channels_location = 10

# random data
array = np.random.rand(batch_size, height, width, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)

# random pixel locations
array_locations = np.random.randint(height*width, size=(batch_size, grid_size, grid_size, grid_size, channels_location))
pixel_locations = tf.convert_to_tensor(array_locations, dtype=tf.int32)

W = weight_variable([3, 3, 3, channels])
h = conv2d(data, W)

[y, top_count, top_voxel_locations] = backprojecting_op.backproject(h, pixel_locations)
y_data = tf.convert_to_tensor(np.ones((batch_size, grid_size, grid_size, grid_size, channels)), dtype=tf.float32)
print y_data, y

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

for step in xrange(10):
    sess.run(train)
    print(step, sess.run(W))
    print(sess.run(y))

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
