TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda

DF_INC_PATH=/var/Projects/FCN/lib/kinect_fusion/include
RD_INC_PATH=/var/Projects/FCN/lib/rendering
RD_LIB_PATH=/var/Projects/FCN/lib/rendering/build

cd matching_loss

g++ -std=c++11 -shared -o matching_loss.so matching_loss_op.cc \
	-I $TF_INC -I $DF_INC_PATH -I $RD_INC_PATH -I $CUDA_PATH/include -fPIC -lrender -L $RD_LIB_PATH -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0

cd ..
echo 'build matching loss'

cd hough_voting_layer

g++ -std=c++11 -c -o Hypothesis.o Hypothesis.cpp -fPIC

g++ -std=c++11 -c -o thread_rand.o thread_rand.cpp -fPIC

g++ -std=c++11 -shared -o hough_voting.so hough_voting_op.cc \
	Hypothesis.o thread_rand.o -I $TF_INC -fPIC -lcudart -lopencv_imgproc -lopencv_calib3d -lopencv_core -lgomp -lnlopt -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0

cd ..
echo 'hough_voting_layer'

cd roi_pooling_layer

nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..
echo 'roi_pooling_layer'
