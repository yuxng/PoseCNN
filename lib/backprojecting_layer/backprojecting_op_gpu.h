#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_BACKPROJECTING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_BACKPROJECTING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool BackprojectForwardLaucher(
    const float* bottom_data, const float* bottom_label,
    const float* bottom_depth, const float* bottom_meta_data, const float* bottom_label_3d,
    const int batch_size, const int height, const int width, const int channels, const int num_classes, const int num_meta_data, 
    const int grid_size, const int kernel_size, const float threshold,
    float* top_data, float* top_label, float* top_flag, const Eigen::GpuDevice& d);

bool BackprojectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, const int grid_size, 
    float* bottom_diff, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
