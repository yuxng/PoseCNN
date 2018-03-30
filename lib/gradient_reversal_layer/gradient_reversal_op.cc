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

// Gradient Reversal Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Gradientreversal")
    .Attr("T: {float, double}")
    .Attr("lambda: float")
    .Input("bottom_data: T")
    .Output("top_data: T");

REGISTER_OP("GradientreversalGrad")
    .Attr("T: {float, double}")
    .Attr("lambda: float")
    .Input("bottom_data: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class GradientreversalOp : public OpKernel {
 public:
  explicit GradientreversalOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the lambda
    OP_REQUIRES_OK(context,
                   context->GetAttr("lambda", &lambda_));
    // Check that lambda is positive
    OP_REQUIRES(context, lambda_ > 0,
                errors::InvalidArgument("Need lambda > 0, got ", lambda_));
  }

  // bottom_data: (batch_size, grid_size, grid_size, grid_size, channels)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();
    int size = bottom_data.NumElements();

    // Create output tensors
    TensorShape output_shape = bottom_data.shape();

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    // identity mapping
    for (int i = 0; i < size; i++)
      top_data(i) = bottom_data_flat(i);
  }
 private:
  float lambda_;
};

REGISTER_KERNEL_BUILDER(Name("Gradientreversal").Device(DEVICE_CPU).TypeConstraint<float>("T"), GradientreversalOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Gradientreversal").Device(DEVICE_CPU).TypeConstraint<double>("T"), GradientreversalOp<CPUDevice, double>);

bool GradientreversalForwardLaucher(const float* bottom_data, const int size, float* top_data, const Eigen::GpuDevice& d);

static void GradientreversalKernel(
    OpKernelContext* context, const Tensor* bottom_data, 
    const int size, const TensorShape& tensor_output_shape) 
{
  Tensor* top_data = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));

  if (!context->status().ok()) {
    return;
  }

  GradientreversalForwardLaucher(
    bottom_data->flat<float>().data(), size, 
    top_data->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class GradientreversalOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit GradientreversalOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the lambda
    OP_REQUIRES_OK(context,
                   context->GetAttr("lambda", &lambda_));
    // Check that lambda is positive
    OP_REQUIRES(context, lambda_ > 0,
                errors::InvalidArgument("Need lambda > 0, got ", lambda_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    int size = bottom_data.NumElements();

    // Create output tensors
    TensorShape output_shape = bottom_data.shape();

    GradientreversalKernel(context, &bottom_data, size, output_shape);
  }
 private:
  float lambda_;
};

REGISTER_KERNEL_BUILDER(Name("Gradientreversal").Device(DEVICE_GPU).TypeConstraint<float>("T"), GradientreversalOp<Eigen::GpuDevice, float>);


bool GradientreversalBackwardLaucher(const float* top_diff, const int size, const float lambda,
    float* bottom_diff, const Eigen::GpuDevice& d);

static void GradientreversalGradKernel(
    OpKernelContext* context, const Tensor* out_backprop,
    const int size, const float lambda, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  GradientreversalBackwardLaucher(
    out_backprop->flat<float>().data(), size, lambda,
    output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class GradientreversalGradOp : public OpKernel {
 public:
  explicit GradientreversalGradOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get the lambda
    OP_REQUIRES_OK(context,
                   context->GetAttr("lambda", &lambda_));
    // Check that lambda is positive
    OP_REQUIRES(context, lambda_ > 0,
                errors::InvalidArgument("Need lambda > 0, got ", lambda_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& out_backprop = context->input(1);

    // size
    int size = bottom_data.NumElements();

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    GradientreversalGradKernel(context, &out_backprop, size, lambda_, output_shape);
  }
 private:
  float lambda_;
};

REGISTER_KERNEL_BUILDER(Name("GradientreversalGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), GradientreversalGradOp<Eigen::GpuDevice, float>);
