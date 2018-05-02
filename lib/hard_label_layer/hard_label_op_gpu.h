#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_HARDLABEL_OP_GPU_H_
#define TENSORFLOW_USER_OPS_HARDLABEL_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool HardlabelForwardLaucher(const float* bottom_prob, const int* bottom_gt,
  const int batch_size, const int height, const int width, const int num_classes,
  const float threshold, float* top_data, const Eigen::GpuDevice& d);

bool HardlabelBackwardLaucher(const float* top_diff, const int batch_size, const int height, const int width, const int num_classes,
    float* bottom_diff_prob, float* bottom_diff_gt, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
