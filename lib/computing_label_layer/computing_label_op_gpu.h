#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_COMPUTING_LABEL_OP_GPU_H_
#define TENSORFLOW_USER_OPS_COMPUTING_LABEL_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool ComputingLabelLaucher(
    const float* bottom_data, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int num_meta_data,
    const int grid_size, const int num_classes, int* top_label, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
