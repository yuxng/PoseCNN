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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Backproject")
    .Attr("T: {float, double}")
    .Input("bottom_data: T")
    .Input("bottom_pixel_locations: int32")
    .Output("top_data: T")
    .Output("top_count: int32")
    .Output("top_voxel_locations: int32");

REGISTER_OP("BackprojectGrad")
    .Attr("T: {float, double}")
    .Input("bottom_data: T")
    .Input("top_count: int32")
    .Input("top_voxel_locations: int32")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class BackprojectOp : public OpKernel {
 public:
  explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  // bottom_data: (batch_size, height, width, channels)
  // bottom_pixel_locations: (batch_size, grid_size, grid_size, grid_size, channels_location)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_pixel_locations = context->input(1);
    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_pixel_locations_flat = bottom_pixel_locations.flat<int>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // pixel location should have 5 dimensions.
    OP_REQUIRES(context, bottom_pixel_locations.dims() == 5,
                errors::InvalidArgument("indexes must be 5-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels_data = bottom_data.dim_size(3);
    int num_channels_location = bottom_pixel_locations.dim_size(4);
    // grid size
    int grid_size = bottom_pixel_locations.dim_size(1);

    // Create output tensors
    // top_data
    int dims[5];
    dims[0] = batch_size;
    dims[1] = grid_size;
    dims[2] = grid_size;
    dims[3] = grid_size;
    dims[4] = num_channels_data;
    TensorShape output_shape_1;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_1);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape_1, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();
    
    // top_count
    TensorShape output_shape_2;
    dims[4] = 1;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_2);

    Tensor* top_count_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_2, &top_count_tensor));
    auto top_count = top_count_tensor->template flat<int>();

    // top_voxel_locations    
    int dims_1[4];
    dims_1[0] = batch_size;
    dims_1[1] = data_height;
    dims_1[2] = data_width;
    dims_1[3] = 1;
    TensorShape output_shape_3;
    TensorShapeUtils::MakeShape(dims_1, 4, &output_shape_3);
    Tensor* top_voxel_locations_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_3, &top_voxel_locations_tensor));
    auto top_voxel_locations = top_voxel_locations_tensor->template flat<int>();

    // Set all element of the voxel location tensor to -1.
    const int N = top_voxel_locations.size();
    for (int i = 0; i < N; i++) 
      top_voxel_locations(i) = -1;

    for(int n = 0; n < batch_size; n++)
    {
      int index_batch_data = n * grid_size * grid_size * grid_size * num_channels_data;
      int index_batch_location = n * grid_size * grid_size * grid_size * num_channels_location;
      for(int d = 0; d < grid_size; d++)
      {
        int index_depth_data = d * grid_size * grid_size * num_channels_data;
        int index_depth_location = d * grid_size * grid_size * num_channels_location;
        for(int h = 0; h < grid_size; h++)
        {
          int index_height_data = h * grid_size * num_channels_data;
          int index_height_location = h * grid_size * num_channels_location;
          for(int w = 0; w < grid_size; w++)
          {
            // set to zero
            for (int c = 0; c < num_channels_data; c++)
            {
              int index_data = w * num_channels_data + c;
              top_data(index_batch_data + index_depth_data + index_height_data + index_data) = 0;
            }

            // find the mapping of this voxel to pixels
            int count = 0;
            for(int c_index = 0; c_index < num_channels_location; c_index++)
            {
              int index_location = w * num_channels_location + c_index;
              int location = bottom_pixel_locations_flat(index_batch_location + index_depth_location + index_height_location + index_location);
              if (location > 0)
              {
                count++;
                int pixel_w = location % data_width;
                int pixel_h = location / data_width;
                for(int c = 0; c < num_channels_data; c++)
                {
                  int index_data = w * num_channels_data + c;
                  int index_pixel = (n * data_height * data_width + pixel_h * data_width + pixel_w) * num_channels_data + c;
                  top_data(index_batch_data + index_depth_data + index_height_data + index_data) += bottom_data_flat(index_pixel);
                }
                // store the voxel location
                top_voxel_locations(n * data_height * data_width + pixel_h * data_width + pixel_w) = d * grid_size * grid_size + h * grid_size + w;
              }
            }
            // compute the mean
            if (count > 1)
            {
              for(int c = 0; c < num_channels_data; c++)
              {
                int index_data = w * num_channels_data + c;
                top_data(index_batch_data + index_depth_data + index_height_data + index_data) /= count;
              }
            }
            // store count
            top_count(n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) = count;
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<float>("T"), BackprojectOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<double>("T"), BackprojectOp<CPUDevice, double>);

bool BackprojectForwardLaucher(
    const float* bottom_data, const int* bottom_pixel_locations, const int batch_size, const int height,
    const int width, const int channels, const int grid_size, const int channels_location,
    float* top_data, int* top_count, int* top_voxel_locations, const Eigen::GpuDevice& d);

static void BackprojectingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_pixel_locations,
    const int batch_size, const int height, const int width, const int channels, const int grid_size, const int channels_location,
    const TensorShape& tensor_output_shape_1, const TensorShape& tensor_output_shape_2, const TensorShape& tensor_output_shape_3) 
{
  Tensor* top_data = nullptr;
  Tensor* top_count = nullptr;
  Tensor* top_voxel_locations = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape_1, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_2, &top_count));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape_3, &top_voxel_locations));

  if (!context->status().ok()) {
    return;
  }

  BackprojectForwardLaucher(
    bottom_data->flat<float>().data(), bottom_pixel_locations->flat<int>().data(),
    batch_size, height, width, channels, grid_size, channels_location,
    top_data->flat<float>().data(), top_count->flat<int>().data(), 
    top_voxel_locations->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class BackprojectOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_pixel_locations = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // pixel location should have 4 dimensions.
    OP_REQUIRES(context, bottom_pixel_locations.dims() == 5,
                errors::InvalidArgument("indexes must be 5-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels_data = bottom_data.dim_size(3);
    int num_channels_location = bottom_pixel_locations.dim_size(4);
    // grid size
    int grid_size = bottom_pixel_locations.dim_size(1);

    // construct the output shape
    int dims[5];
    dims[0] = batch_size;
    dims[1] = grid_size;
    dims[2] = grid_size;
    dims[3] = grid_size;
    dims[4] = num_channels_data;
    TensorShape output_shape_1;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_1);
    
    TensorShape output_shape_2;
    dims[4] = 1;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_2);
    
    int dims_1[4];
    dims_1[0] = batch_size;
    dims_1[1] = data_height;
    dims_1[2] = data_width;
    dims_1[3] = 1;
    TensorShape output_shape_3;
    TensorShapeUtils::MakeShape(dims_1, 4, &output_shape_3);

    BackprojectingKernel(context, &bottom_data, &bottom_pixel_locations, batch_size, data_height,
      data_width, num_channels_data, grid_size, num_channels_location, output_shape_1, output_shape_2, output_shape_3);
  }
};

REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_GPU).TypeConstraint<float>("T"), BackprojectOp<Eigen::GpuDevice, float>);


bool BackprojectBackwardLaucher(const float* top_diff, const int* top_count, const int* top_voxel_locations, const int batch_size,
    const int height, const int width, const int channels, const int grid_size,
    float* bottom_diff, const Eigen::GpuDevice& d);

static void BackprojectingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* top_count, const Tensor* top_voxel_locations, const Tensor* out_backprop,
    const int batch_size, const int height, const int width, const int channels, const int grid_size, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  BackprojectBackwardLaucher(
    out_backprop->flat<float>().data(), top_count->flat<int>().data(), top_voxel_locations->flat<int>().data(),
    batch_size, height, width, channels, grid_size, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class BackprojectGradOp : public OpKernel {
 public:
  explicit BackprojectGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& top_count = context->input(1);
    const Tensor& top_voxel_locations = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, top_count.dims() == 5,
                errors::InvalidArgument("count must be 5-dimensional"));

    OP_REQUIRES(context, top_voxel_locations.dims() == 4,
                errors::InvalidArgument("voxel_locations must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(1);
    // data width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);
    // grid size
    int grid_size = top_count.dim_size(1);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    BackprojectingGradKernel(
      context, &bottom_data, &top_count, &top_voxel_locations, &out_backprop,
      batch_size, height, width, channels, grid_size, output_shape);

  }
};

REGISTER_KERNEL_BUILDER(Name("BackprojectGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), BackprojectGradOp<Eigen::GpuDevice, float>);
