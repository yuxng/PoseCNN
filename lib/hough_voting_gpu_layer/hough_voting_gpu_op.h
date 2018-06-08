#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_HOUGHVOTING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_HOUGHVOTING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

void HoughVotingLaucher(OpKernelContext* context,
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
    const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
    const int is_train, const float inlierThreshold, const float votingThreshold, const float perThreshold,
    float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, const Eigen::GpuDevice& d);

void reset_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, int num_classes);

void copy_num_rois(int* num_rois, int* num_rois_device);

void copy_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain,
  float* top_box_final, float* top_pose_final, float* top_target_final, float* top_weight_final, int* top_domain_final, int num_classes, int num_rois);

void set_gradients(float* top_label, float* top_vertex, int batch_size, int height, int width, int num_classes);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
