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

#cd hard_label_layer

#nvcc -std=c++11 -c -o hard_label_op.cu.o hard_label_op_gpu.cu.cc \
#	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o hard_label.so hard_label_op.cc \
#	hard_label_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -lcublas -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

#cd ..
#echo 'hard_label_layer'

#cd gradient_reversal_layer

#nvcc -std=c++11 -c -o gradient_reversal_op.cu.o gradient_reversal_op_gpu.cu.cc \
#	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o gradient_reversal.so gradient_reversal_op.cc \
#	gradient_reversal_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -lcublas -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

#cd ..
#echo 'gradient_reversal_layer'

#cd hough_voting_gpu_layer

#nvcc -std=c++11 -c -o hough_voting_gpu_op.cu.o hough_voting_gpu_op.cu.cc \
#	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o hough_voting_gpu.so hough_voting_gpu_op.cc \
#	hough_voting_gpu_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -lcublas -lopencv_imgproc -lopencv_calib3d -lopencv_core -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework

#cd ..
#echo 'hough_voting_gpu_layer'
