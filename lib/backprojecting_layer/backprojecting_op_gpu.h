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
    const float* bottom_data, const int* bottom_pixel_locations,
    const int batch_size, const int height, const int width, const int channels,
    const int grid_size, const int channels_location,
    float* top_data, int* top_count, int* top_voxel_locations, const Eigen::GpuDevice& d);

bool BackprojectBackwardLaucher(const float* top_diff, const int* top_count, const int* top_voxel_locations, 
    const int batch_size, const int height, const int width, const int channels, const int grid_size,
    float* bottom_diff, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
