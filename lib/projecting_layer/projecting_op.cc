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
    .Input("bottom_data: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Output("top_data: T");

template <typename Device, typename T>
class ProjectOp : public OpKernel {
 public:
  explicit ProjectOp(OpKernelConstruction* context) : OpKernel(context) {
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
    // projection matrix: meta_data[0 ~ 11]
    // camera center: meta_data[12, 13, 14]
    // voxel step size: meta_data[15, 16, 17]
    // voxel min value: meta_data[18, 19, 20]
    // backprojection matrix: meta_data[21 ~ 32]
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
          // projection matrix: meta_data[0 ~ 11]
          // camera center: meta_data[12, 13, 14]
          // voxel step size: meta_data[15, 16, 17]
          // voxel min value: meta_data[18, 19, 20]
          // backprojection matrix: meta_data[21 ~ 32]
          int offset = n * num_meta_data + 21;
          T X = meta_data(offset + 0) * w + meta_data(offset + 1) * h + meta_data(offset + 2);
          T Y = meta_data(offset + 3) * w + meta_data(offset + 4) * h + meta_data(offset + 5);
          T Z = meta_data(offset + 6) * w + meta_data(offset + 7) * h + meta_data(offset + 8);
          T W = meta_data(offset + 9) * w + meta_data(offset + 10) * h + meta_data(offset + 11);
          X /= W;
          Y /= W;
          Z /= W;

          // compute the ray
          offset = n * num_meta_data;
          T RX = X - meta_data(offset + 12);
          T RY = Y - meta_data(offset + 13);
          T RZ = Z - meta_data(offset + 14);

          // compute the norm
          T N = sqrt(RX*RX + RY*RY + RZ*RZ);
        
          // normalization
          RX /= N;
          RY /= N;
          RZ /= N;

          // compute the 3D points
          X = meta_data(offset + 12) + depth * RX;
          Y = meta_data(offset + 13) + depth * RY;
          Z = meta_data(offset + 14) + depth * RZ;

          // voxel location in 3D
          int vd = floor((X - meta_data(offset + 18)) / meta_data(offset + 15));
          int vh = floor((Y - meta_data(offset + 19)) / meta_data(offset + 16));
          int vw = floor((Z - meta_data(offset + 20)) / meta_data(offset + 17));

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
};

REGISTER_KERNEL_BUILDER(Name("Project").Device(DEVICE_GPU).TypeConstraint<float>("T"), ProjectOp<Eigen::GpuDevice, float>);
