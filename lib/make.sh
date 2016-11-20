TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda

cd backprojecting_layer

nvcc -std=c++11 -c -o backprojecting_op.cu.o backprojecting_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o backprojecting.so backprojecting_op.cc \
	backprojecting_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
echo 'build backprojecting layer'

cd projecting_layer

nvcc -std=c++11 -c -o projecting_op.cu.o projecting_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o projecting.so projecting_op.cc \
	projecting_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
echo 'build projecting layer'

cd computing_label_layer

nvcc -std=c++11 -c -o computing_label_op.cu.o computing_label_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o computing_label.so computing_label_op.cc \
	computing_label_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
echo 'build computing label layer'
