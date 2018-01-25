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

cd hough_voting_gpu_layer

nvcc -std=c++11 -c -o hough_voting_gpu_op.cu.o hough_voting_gpu_op.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o hough_voting_gpu.so hough_voting_gpu_op.cc \
	hough_voting_gpu_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -lcublas -lopencv_imgproc -lopencv_calib3d -lopencv_core -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

cd ..
echo 'hough_voting_gpu_layer'

cd hough_voting_layer

g++ -std=c++11 -c -o Hypothesis.o Hypothesis.cpp -fPIC

g++ -std=c++11 -c -o thread_rand.o thread_rand.cpp -fPIC

g++ -std=c++11 -shared -o hough_voting.so hough_voting_op.cc \
	Hypothesis.o thread_rand.o -I $TF_INC -I$TF_INC/external/nsync/public \
        -fPIC -lcudart -lopencv_imgproc -lopencv_calib3d -lopencv_core -lgomp -lnlopt -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

cd ..
echo 'hough_voting_layer'

cd roi_pooling_layer

nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
	roi_pooling_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'roi_pooling_layer'

cd triplet_loss

nvcc -std=c++11 -c -o triplet_loss_op.cu.o triplet_loss_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o triplet_loss.so triplet_loss_op.cc \
	triplet_loss_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build triplet loss'

cd lifted_structured_loss

nvcc -std=c++11 -c -o lifted_structured_loss_op.cu.o lifted_structured_loss_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o lifted_structured_loss.so lifted_structured_loss_op.cc \
	lifted_structured_loss_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build lifted structured loss'

cd computing_flow_layer

nvcc -std=c++11 -c -o computing_flow_op.cu.o computing_flow_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o computing_flow.so computing_flow_op.cc \
	computing_flow_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build computing flow layer'

cd backprojecting_layer

nvcc -std=c++11 -c -o backprojecting_op.cu.o backprojecting_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o backprojecting.so backprojecting_op.cc \
	backprojecting_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build backprojecting layer'

cd projecting_layer

nvcc -std=c++11 -c -o projecting_op.cu.o projecting_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o projecting.so projecting_op.cc \
	projecting_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build projecting layer'

cd computing_label_layer

nvcc -std=c++11 -c -o computing_label_op.cu.o computing_label_op_gpu.cu.cc \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o computing_label.so computing_label_op.cc \
	computing_label_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework
cd ..
echo 'build computing label layer'
