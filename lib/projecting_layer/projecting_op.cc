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

// Projecting Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Project")
    .Attr("T: {float, double}")
    .Attr("kernel_size: int")
    .Attr("threshold: float")
    .Input("bottom_data: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Output("top_data: T");

REGISTER_OP("ProjectGrad")
    .Attr("T: {float, double}")
    .Attr("kernel_size: int")
    .Attr("threshold: float")
    .Input("bottom_data: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class ProjectOp : public OpKernel {
 public:
  explicit ProjectOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the kernel size
    OP_REQUIRES_OK(context,
                   context->GetAttr("kernel_size", &kernel_size_));
    // Check that kernel size is positive
    OP_REQUIRES(context, kernel_size_ >= 0,
                errors::InvalidArgument("Need kernel_size >= 0, got ", kernel_size_));
    // Get the threshold
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold", &threshold_));
    // Check that threshold is positive
    OP_REQUIRES(context, threshold_ >= 0,
                errors::InvalidArgument("Need threshold >= 0, got ", threshold_));
  }

  // bottom_data: (batch_size, grid_size, grid_size, grid_size, channels)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();

    const Tensor& bottom_depth = context->input(1);
    auto im_depth = bottom_depth.flat<T>();

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Tensor& bottom_meta_data = context->input(2);
    auto meta_data = bottom_meta_data.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 5,
                errors::InvalidArgument("data must be 5-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // grid size
    int grid_size = bottom_data.dim_size(1);
    // number of channels
    int num_channels = bottom_data.dim_size(4);
    // height
    int height = bottom_depth.dim_size(1);
    // width
    int width = bottom_depth.dim_size(2);
    // number of meta data
    int num_meta_data = bottom_meta_data.dim_size(3);

    // Create output tensors
    // top_data
    int dims[4];
    dims[0] = batch_size;
    dims[1] = height;
    dims[2] = width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    int index_meta_data = 0;    
    for(int n = 0; n < batch_size; n++)
    {
      for(int h = 0; h < height; h++)
      {
        for(int w = 0; w < width; w++)
        {
          int index_pixel = n * height * width + h * width + w;
          T depth = im_depth(index_pixel);

          // find the voxel for this pixel

          // backproject the pixel to 3D
          // format of the meta_data
          // intrinsic matrix: meta_data[0 ~ 8]
          // inverse intrinsic matrix: meta_data[9 ~ 17]
          // pose_world2live: meta_data[18 ~ 29]
          // pose_live2world: meta_data[30 ~ 41]
          // voxel step size: meta_data[42, 43, 44]
          // voxel min value: meta_data[45, 46, 47]

          // apply the inverse intrinsic matrix
          int offset = n * num_meta_data + 9;
          T RX = meta_data(offset + 0) * w + meta_data(offset + 1) * h + meta_data(offset + 2);
          T RY = meta_data(offset + 3) * w + meta_data(offset + 4) * h + meta_data(offset + 5);
          T RZ = meta_data(offset + 6) * w + meta_data(offset + 7) * h + meta_data(offset + 8);

          // compute the 3D points in camera's coordinate system
          T X = depth * RX;
          T Y = depth * RY;
          T Z = depth * RZ;

          // apply pose_live2world
          offset = n * num_meta_data;
          T X1 = meta_data(offset + 30) * X + meta_data(offset + 31) * Y + meta_data(offset + 32) * Z + meta_data(offset + 33);
          T Y1 = meta_data(offset + 34) * X + meta_data(offset + 35) * Y + meta_data(offset + 36) * Z + meta_data(offset + 37);
          T Z1 = meta_data(offset + 38) * X + meta_data(offset + 39) * Y + meta_data(offset + 40) * Z + meta_data(offset + 41);

          // voxel location in 3D
          int vd = round((X1 - meta_data(offset + 45)) / meta_data(offset + 42));
          int vh = round((Y1 - meta_data(offset + 46)) / meta_data(offset + 43));
          int vw = round((Z1 - meta_data(offset + 47)) / meta_data(offset + 44));

          // get the data
          if (vd >= 0 && vd < grid_size && vh >= 0 && vh < grid_size && vw >= 0 && vw < grid_size)
          {
            for(int c = 0; c < num_channels; c++)
              top_data(index_pixel * num_channels + c) = bottom_data_flat((n * grid_size * grid_size * grid_size + vd * grid_size * grid_size + vh * grid_size + vw) * num_channels + c);
          }
          else
          {
            for(int c = 0; c < num_channels; c++)
              top_data(index_pixel * num_channels + c) = 0;
          }
        }
      }
    }
  }
 private:
  int kernel_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Project").Device(DEVICE_CPU).TypeConstraint<float>("T"), ProjectOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Project").Device(DEVICE_CPU).TypeConstraint<double>("T"), ProjectOp<CPUDevice, double>);

bool ProjectForwardLaucher(
    const float* bottom_data, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, float* top_data, const Eigen::GpuDevice& d);

static void ProjectingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_depth, const Tensor* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, 
    const int grid_size, const TensorShape& tensor_output_shape) 
{
  Tensor* top_data = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));

  if (!context->status().ok()) {
    return;
  }

  ProjectForwardLaucher(
    bottom_data->flat<float>().data(), bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, grid_size,
    top_data->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class ProjectOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit ProjectOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the kernel size
    OP_REQUIRES_OK(context,
                   context->GetAttr("kernel_size", &kernel_size_));
    // Check that kernel size is positive
    OP_REQUIRES(context, kernel_size_ >= 0,
                errors::InvalidArgument("Need kernel_size >= 0, got ", kernel_size_));
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

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 5,
                errors::InvalidArgument("data must be 5-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // grid size
    int grid_size = bottom_data.dim_size(1);
    // number of channels
    int num_channels = bottom_data.dim_size(4);
    // height
    int height = bottom_depth.dim_size(1);
    // width
    int width = bottom_depth.dim_size(2);
    // number of meta data
    int num_meta_data = bottom_meta_data.dim_size(3);

    // Create output tensors
    // top_data
    int dims[4];
    dims[0] = batch_size;
    dims[1] = height;
    dims[2] = width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    ProjectingKernel(context, &bottom_data, &bottom_depth, &bottom_meta_data, batch_size, height,
      width, num_channels, num_meta_data, grid_size, output_shape);
  }
 private:
  int kernel_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Project").Device(DEVICE_GPU).TypeConstraint<float>("T"), ProjectOp<Eigen::GpuDevice, float>);


bool ProjectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int grid_size, const int kernel_size, const float threshold,
    float* bottom_diff, const Eigen::GpuDevice& d);

static void ProjectingGradKernel(
    OpKernelContext* context, const Tensor* bottom_depth, const Tensor* bottom_meta_data, const Tensor* out_backprop,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, 
    const int grid_size, const int kernel_size, const float threshold,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ProjectBackwardLaucher(
    out_backprop->flat<float>().data(), bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, grid_size, kernel_size, threshold, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class ProjectGradOp : public OpKernel {
 public:
  explicit ProjectGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the kernel size
    OP_REQUIRES_OK(context,
                   context->GetAttr("kernel_size", &kernel_size_));
    // Check that kernel size is positive
    OP_REQUIRES(context, kernel_size_ >= 0,
                errors::InvalidArgument("Need kernel_size >= 0, got ", kernel_size_));
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

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 5,
                errors::InvalidArgument("data must be 5-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // grid size
    int grid_size = bottom_data.dim_size(1);
    // number of channels
    int num_channels = bottom_data.dim_size(4);
    // height
    int height = bottom_depth.dim_size(1);
    // width
    int width = bottom_depth.dim_size(2);
    // number of meta data
    int num_meta_data = bottom_meta_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    ProjectingGradKernel(
      context, &bottom_depth, &bottom_meta_data, &out_backprop,
      batch_size, height, width, num_channels, num_meta_data, grid_size, kernel_size_, threshold_, output_shape);

  }
 private:
  int kernel_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("ProjectGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), ProjectGradOp<Eigen::GpuDevice, float>);
