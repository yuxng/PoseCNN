#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_AVERAGE_DISTANCE_OP_GPU_H_
#define TENSORFLOW_USER_OPS_AVERAGE_DISTANCE_OP_GPU_H_

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
void AveragedistanceForwardLaucher(OpKernelContext* context,
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const float* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);

bool AveragedistanceBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int channels, float* output, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
