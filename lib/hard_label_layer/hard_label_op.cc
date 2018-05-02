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

// Hard Label Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Hardlabel")
    .Attr("T: {float, double}")
    .Attr("threshold: float")
    .Input("bottom_prob: T")
    .Input("bottom_gt: int32")
    .Output("top_data: T");

REGISTER_OP("HardlabelGrad")
    .Attr("T: {float, double}")
    .Attr("threshold: float")
    .Input("bottom_prob: T")
    .Input("bottom_gt: int32")
    .Input("grad: T")
    .Output("output_prob: T")
    .Output("output_gt: T");

template <typename Device, typename T>
class HardlabelOp : public OpKernel {
 public:
  explicit HardlabelOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ > 0,
                errors::InvalidArgument("Need threshold > 0, got ", threshold_));
  }

  // bottom_prob: (batch_size, height, width, num_classes)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prob = context->input(0);
    auto bottom_prob_flat = bottom_prob.flat<T>();

    const Tensor& bottom_gt = context->input(1);
    auto bottom_gt_flat = bottom_gt.flat<int>();

    OP_REQUIRES(context, bottom_prob.dims() == 4,
                errors::InvalidArgument("prob must be 4-dimensional"));

    OP_REQUIRES(context, bottom_gt.dims() == 3,
                errors::InvalidArgument("gt label must be 3-dimensional"));

    // batch size
    int batch_size = bottom_prob.dim_size(0);
    // height
    int height = bottom_prob.dim_size(1);
    // width
    int width = bottom_prob.dim_size(2);
    // number of classes
    int num_classes = bottom_prob.dim_size(3);

    // Create output tensors
    int dims[4];
    dims[0] = batch_size;
    dims[1] = height;
    dims[2] = width;
    dims[3] = num_classes;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    for(int n = 0; n < batch_size; n++)
    {
      for(int h = 0; h < height; h++)
      {
        for(int w = 0; w < width; w++)
        {
          int index_pixel = n * height * width + h * width + w;
          int label_gt = bottom_gt_flat(index_pixel);
          T prob = bottom_prob_flat(index_pixel * num_classes + label_gt);

          for (int c = 0; c < num_classes; c++)
            top_data(index_pixel * num_classes + c) = 0.0;

          if (prob < threshold_)
            top_data(index_pixel * num_classes + label_gt) = 1.0;
        }
      }
    }
  }
 private:
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Hardlabel").Device(DEVICE_CPU).TypeConstraint<float>("T"), HardlabelOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Hardlabel").Device(DEVICE_CPU).TypeConstraint<double>("T"), HardlabelOp<CPUDevice, double>);

bool HardlabelForwardLaucher(const float* bottom_prob, const int* bottom_gt,
  const int batch_size, const int height, const int width, const int num_classes,
  const float threshold, float* top_data, const Eigen::GpuDevice& d);

static void HardlabelKernel(
    OpKernelContext* context, const Tensor* bottom_prob, const Tensor* bottom_gt,
    const int batch_size, const int height, const int width, const int num_classes,
    const float threshold, const TensorShape& tensor_output_shape) 
{
  Tensor* top_data = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));

  if (!context->status().ok()) {
    return;
  }

  HardlabelForwardLaucher(
    bottom_prob->flat<float>().data(), bottom_gt->flat<int>().data(), batch_size, height, width, num_classes, threshold, 
    top_data->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class HardlabelOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit HardlabelOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ > 0,
                errors::InvalidArgument("Need threshold > 0, got ", threshold_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prob = context->input(0);
    const Tensor& bottom_gt = context->input(1);

    // batch size
    int batch_size = bottom_prob.dim_size(0);
    // height
    int height = bottom_prob.dim_size(1);
    // width
    int width = bottom_prob.dim_size(2);
    // number of classes
    int num_classes = bottom_prob.dim_size(3);

    // Create output tensors
    int dims[4];
    dims[0] = batch_size;
    dims[1] = height;
    dims[2] = width;
    dims[3] = num_classes;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    HardlabelKernel(context, &bottom_prob, &bottom_gt, batch_size, height, width, num_classes, threshold_, output_shape);
  }
 private:
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Hardlabel").Device(DEVICE_GPU).TypeConstraint<float>("T"), HardlabelOp<Eigen::GpuDevice, float>);


bool HardlabelBackwardLaucher(const float* top_diff, const int batch_size, const int height, const int width, const int num_classes,
    float* bottom_diff_prob, float* bottom_diff_gt, const Eigen::GpuDevice& d);

static void HardlabelGradKernel(
    OpKernelContext* context, const Tensor* out_backprop, const int batch_size, const int height, const int width, const int num_classes,
    const TensorShape& tensor_output_shape_prob, const TensorShape& tensor_output_shape_gt) 
{
  Tensor* output_prob = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape_prob, &output_prob));

  Tensor* output_gt = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_gt, &output_gt));

  if (!context->status().ok()) {
    return;
  }

  HardlabelBackwardLaucher(
    out_backprop->flat<float>().data(), batch_size, height, width, num_classes,
    output_prob->flat<float>().data(), output_gt->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class HardlabelGradOp : public OpKernel {
 public:
  explicit HardlabelGradOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ > 0,
                errors::InvalidArgument("Need threshold > 0, got ", threshold_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prob = context->input(0);
    const Tensor& bottom_gt = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // batch size
    int batch_size = bottom_prob.dim_size(0);
    // height
    int height = bottom_prob.dim_size(1);
    // width
    int width = bottom_prob.dim_size(2);
    // number of classes
    int num_classes = bottom_prob.dim_size(3);

    // construct the output shape
    TensorShape output_shape_prob = bottom_prob.shape();
    TensorShape output_shape_gt = bottom_gt.shape();

    HardlabelGradKernel(context, &out_backprop, batch_size, height, width, num_classes, output_shape_prob, output_shape_gt);
  }
 private:
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("HardlabelGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), HardlabelGradOp<Eigen::GpuDevice, float>);
