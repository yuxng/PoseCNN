#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_PROJECTING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_PROJECTING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool ProjectForwardLaucher(
    const float* bottom_data, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, float* top_data, const Eigen::GpuDevice& d);

bool ProjectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int grid_size, const int kernel_size, const float threshold,
    float* bottom_diff, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
