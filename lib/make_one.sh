TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

CUDA_PATH=/usr/local/cuda

cd average_distance_loss

nvcc -std=c++11 -c -o average_distance_loss_op_gpu.cu.o average_distance_loss_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o average_distance_loss.so average_distance_loss_op.cc \
	average_distance_loss_op_gpu.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

cd ..
echo 'average_distance_loss'
