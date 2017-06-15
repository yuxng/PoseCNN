/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Matching Loss Op
#include "rendering.hpp"

static Render render_;

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"


using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Matching")
    .Attr("T: {float, double}")
    .Attr("filename_camera: string")
    .Attr("filename_model: string")
    .Input("bottom_pose: T")
    .Input("bottom_init: T")
    .Input("bottom_gt: T")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("bottom_labels: int32")
    .Output("loss: T")
    .Output("bottom_diff: T");

REGISTER_OP("MatchingGrad")
    .Attr("T: {float, double}")
    .Input("bottom_diff: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class MatchingOp : public OpKernel {
 public:
  explicit MatchingOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the model filename
    OP_REQUIRES_OK(context, context->GetAttr("filename_model", &filename_model_));
    std::cout << filename_model_ << std::endl;

    if (!pangolin::FileExists(filename_model_))
    {
        throw std::runtime_error("3D model filename does not exist!");
    }

    // Get the camera filename
    OP_REQUIRES_OK(context, context->GetAttr("filename_camera", &filename_camera_));
    std::cout << filename_camera_ << std::endl;

    if (!pangolin::FileExists(filename_camera_)) 
    {
        throw std::runtime_error("Camera filename does not exist!");
    }

    render_.setup(filename_camera_, filename_model_);
  }

  // bottom_pose: (batch_size, 4 * num_classes)
  // bottom_gt: (batch_size, 13)
  void Compute(OpKernelContext* context) override 
  {

    // Grab the input tensor
    const Tensor& bottom_pose = context->input(0);
    const T* pose = bottom_pose.flat<T>().data();

    const Tensor& bottom_init = context->input(1);
    const T* init = bottom_init.flat<T>().data();

    const Tensor& bottom_gt = context->input(2);
    const T* gt = bottom_gt.flat<T>().data();
    
    const Tensor& bottom_data = context->input(3);
    const T* data = bottom_data.flat<T>().data();

    const Tensor& bottom_rois = context->input(4);
    const T* rois = bottom_rois.flat<T>().data();

    const Tensor& bottom_labels = context->input(5);
    const int* labels = bottom_labels.flat<int>().data();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_pose.dims() == 2,
                errors::InvalidArgument("pose must be 2-dimensional"));

    OP_REQUIRES(context, bottom_gt.dims() == 2,
                errors::InvalidArgument("gt must be 2-dimensional"));

    // batch size
    int batch_size = bottom_pose.dim_size(0);
    // number of channels
    int num_channels = bottom_pose.dim_size(1);
    int num_classes = num_channels / 4;
    // gt size
    int gt_size = bottom_gt.dim_size(0);

    int num_images = bottom_data.dim_size(0);
    int height = bottom_data.dim_size(1);
    int width = bottom_data.dim_size(2);
    int num_rois = bottom_rois.dim_size(0);

    // Create output loss tensor
    int dim = 1;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    // bottom diff
    TensorShape output_shape_diff = bottom_pose.shape();
    Tensor* bottom_diff_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_diff, &bottom_diff_tensor));
    T* bottom_diff = bottom_diff_tensor->template flat<T>().data();
    memset(bottom_diff, 0, batch_size * num_channels *sizeof(T));

    // rendering to compute the loss
    T loss = render_.render(data, labels, rois, num_rois, gt_size, num_classes, gt, pose, init, bottom_diff);

    top_data(0) = loss;
  }
 private:
  // file names
  std::string filename_model_;
  std::string filename_camera_;
};

REGISTER_KERNEL_BUILDER(Name("Matching").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatchingOp<CPUDevice, float>);

// compute gradient
template <class Device, class T>
class MatchingGradOp : public OpKernel {
 public:
  explicit MatchingGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {
  }

  void Compute(OpKernelContext* context) override 
  {
    const Tensor& bottom_diff = context->input(0);
    auto bottom_diff_flat = bottom_diff.flat<T>();
    
    const Tensor& out_backprop = context->input(1);
    T loss = out_backprop.flat<T>()(0);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_diff.dims() == 2,
                errors::InvalidArgument("bottom diff must be 2-dimensional"));

    // batch size
    int batch_size = bottom_diff.dim_size(0);
    // number of channels
    int num_channels = bottom_diff.dim_size(1);

    // construct the output shape
    TensorShape output_shape = bottom_diff.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto top_data = output->template flat<T>();

    for (int i = 0; i < batch_size * num_channels; i++)
      top_data(i) = loss * bottom_diff_flat(i);
  }
};

REGISTER_KERNEL_BUILDER(Name("MatchingGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatchingGradOp<CPUDevice, float>);
