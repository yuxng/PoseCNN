#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_GRADIENTREVERSAL_OP_GPU_H_
#define TENSORFLOW_USER_OPS_GRADIENTREVERSAL_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool GradientreversalForwardLaucher(
    const float* bottom_data, const int batch_size, const int height, const int width, const int channels,
    float* top_data, const Eigen::GpuDevice& d);

bool GradientreversalBackwardLaucher(const float* top_diff, const int batch_size,
    const int height, const int width, const int channels, const float lambda,
    float* bottom_diff, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
