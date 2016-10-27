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

// Backprojecting Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Backproject")
    .Attr("T: {float, double}")
    .Attr("grid_size: int")
    .Attr("threshold: float")
    .Input("bottom_data: T")
    .Input("bottom_data_3d: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Output("top_data: T");

REGISTER_OP("BackprojectGrad")
    .Attr("T: {float, double}")
    .Attr("grid_size: int")
    .Attr("threshold: float")
    .Input("bottom_data: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class BackprojectOp : public OpKernel {
 public:
  explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the grid size
    OP_REQUIRES_OK(context,
                   context->GetAttr("grid_size", &grid_size_));
    // Check that grid size is positive
    OP_REQUIRES(context, grid_size_ >= 0,
                errors::InvalidArgument("Need grid_size >= 0, got ", grid_size_));
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ >= 0,
                errors::InvalidArgument("Need threshold >= 0, got ", threshold_));
  }

  // bottom_data: (batch_size, height, width, channels)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();

    const Tensor& bottom_data_3d = context->input(1);
    auto bottom_data_3d_flat = bottom_data_3d.flat<T>();

    const Tensor& bottom_depth = context->input(2);
    auto im_depth = bottom_depth.flat<T>();

    // format of the meta_data
    // projection matrix: meta_data[0 ~ 11]
    // camera center: meta_data[12, 13, 14]
    // voxel step size: meta_data[15, 16, 17]
    // voxel min value: meta_data[18, 19, 20]
    // backprojection matrix: meta_data[21 ~ 32]
    const Tensor& bottom_meta_data = context->input(3);
    auto meta_data = bottom_meta_data.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_data_3d.dims() == 5,
                errors::InvalidArgument("data 3D must be 4-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_meta_data = bottom_meta_data.dim_size(3);

    // Create output tensors
    // top_data
    int dims[5];
    dims[0] = batch_size;
    dims[1] = grid_size_;
    dims[2] = grid_size_;
    dims[3] = grid_size_;
    dims[4] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    int index_meta_data = 0;    
    for(int n = 0; n < batch_size; n++)
    {
      int index_batch = n * grid_size_ * grid_size_ * grid_size_;
      for(int d = 0; d < grid_size_; d++)
      {
        int index_depth = d * grid_size_ * grid_size_;
        for(int h = 0; h < grid_size_; h++)
        {
          int index_height = h * grid_size_;
          for(int w = 0; w < grid_size_; w++)
          {
            // voxel location in 3D
            float X = d * meta_data(index_meta_data + 15) + meta_data(index_meta_data + 18);
            float Y = h * meta_data(index_meta_data + 16) + meta_data(index_meta_data + 19);
            float Z = w * meta_data(index_meta_data + 17) + meta_data(index_meta_data + 20);

            // project the 3D point to image
            float x1 = meta_data(index_meta_data + 0) * X + meta_data(index_meta_data + 1) * Y + meta_data(index_meta_data + 2) * Z + meta_data(index_meta_data + 3);
            float x2 = meta_data(index_meta_data + 4) * X + meta_data(index_meta_data + 5) * Y + meta_data(index_meta_data + 6) * Z + meta_data(index_meta_data + 7);
            float x3 = meta_data(index_meta_data + 8) * X + meta_data(index_meta_data + 9) * Y + meta_data(index_meta_data + 10) * Z + meta_data(index_meta_data + 11);
            int px = round(x1 / x3);
            int py = round(x2 / x3);

            int flag = 0;
            if (px >= 0 && px < width && py >= 0 && py < height)
            {
              int index_pixel = n * height * width + py * width + px;
              T depth = im_depth(index_pixel);
              // distance of this voxel to camera center
              float dvoxel = sqrt((X - meta_data(index_meta_data + 12)) * (X - meta_data(index_meta_data + 12)) 
                                + (Y - meta_data(index_meta_data + 13)) * (Y - meta_data(index_meta_data + 13)) 
                                + (Z - meta_data(index_meta_data + 14)) * (Z - meta_data(index_meta_data + 14)));
              // check if the voxel is on the surface
              if (fabs(depth - dvoxel) < threshold_)
              {
                flag = 1;
                // data
                for(int c = 0; c < num_channels; c++)
                  top_data((index_batch + index_depth + index_height + w) * num_channels + c) = bottom_data_flat(index_pixel * num_channels + c);
              }
            }
            if (flag == 0)
            {
              for (int c = 0; c < num_channels; c++)
                top_data((index_batch + index_depth + index_height + w) * num_channels + c) = bottom_data_3d_flat((index_batch + index_depth + index_height + w) * num_channels + c);
            }
          }
        }
      }
      index_meta_data += num_meta_data;
    }
  }
 private:
  int grid_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<float>("T"), BackprojectOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<double>("T"), BackprojectOp<CPUDevice, double>);

bool BackprojectForwardLaucher(
    const float* bottom_data, const float* bottom_data_3d,
    const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, const float threshold,
    float* top_data, const Eigen::GpuDevice& d);

static void BackprojectingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_data_3d,
    const Tensor* bottom_depth, const Tensor* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, 
    const int grid_size, const float threshold, const TensorShape& tensor_output_shape) 
{
  Tensor* top_data = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));

  if (!context->status().ok()) {
    return;
  }

  BackprojectForwardLaucher(
    bottom_data->flat<float>().data(), bottom_data_3d->flat<float>().data(),
    bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, grid_size, threshold,
    top_data->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class BackprojectOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the grid size
    OP_REQUIRES_OK(context,
                   context->GetAttr("grid_size", &grid_size_));
    // Check that grid size is positive
    OP_REQUIRES(context, grid_size_ >= 0,
                errors::InvalidArgument("Need grid_size >= 0, got ", grid_size_));
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ >= 0,
                errors::InvalidArgument("Need threshold >= 0, got ", threshold_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_data_3d = context->input(1);
    const Tensor& bottom_depth = context->input(2);
    const Tensor& bottom_meta_data = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_data_3d.dims() == 5,
                errors::InvalidArgument("data 3D must be 5-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_meta_data = bottom_meta_data.dim_size(3);

    // Create output tensors
    // top_data
    int dims[5];
    dims[0] = batch_size;
    dims[1] = grid_size_;
    dims[2] = grid_size_;
    dims[3] = grid_size_;
    dims[4] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape);

    BackprojectingKernel(context, &bottom_data, &bottom_data_3d, &bottom_depth, &bottom_meta_data, batch_size, height,
      width, num_channels, num_meta_data, grid_size_, threshold_, output_shape);
  }
 private:
  int grid_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_GPU).TypeConstraint<float>("T"), BackprojectOp<Eigen::GpuDevice, float>);


bool BackprojectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int grid_size,
    float* bottom_diff, const Eigen::GpuDevice& d);

static void BackprojectingGradKernel(
    OpKernelContext* context, const Tensor* bottom_depth, const Tensor* bottom_meta_data, const Tensor* out_backprop,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, const int grid_size,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  BackprojectBackwardLaucher(
    out_backprop->flat<float>().data(), bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, grid_size, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class BackprojectGradOp : public OpKernel {
 public:
  explicit BackprojectGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the grid size
    OP_REQUIRES_OK(context,
                   context->GetAttr("grid_size", &grid_size_));
    // Check that grid size is positive
    OP_REQUIRES(context, grid_size_ >= 0,
                errors::InvalidArgument("Need grid_size >= 0, got ", grid_size_));
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ >= 0,
                errors::InvalidArgument("Need threshold >= 0, got ", threshold_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_depth = context->input(1);
    const Tensor& bottom_meta_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_meta_data = bottom_meta_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    BackprojectingGradKernel(
      context, &bottom_depth, &bottom_meta_data, &out_backprop,
      batch_size, height, width, num_channels, num_meta_data, grid_size_, output_shape);

  }
 private:
  int grid_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("BackprojectGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), BackprojectGradOp<Eigen::GpuDevice, float>);
