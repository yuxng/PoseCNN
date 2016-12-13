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
    .Attr("kernel_size: int")
    .Attr("threshold: float")
    .Input("bottom_data: T")
    .Input("bottom_label: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Input("bottom_label_3d: T")
    .Output("top_data: T")
    .Output("top_label: T")
    .Output("top_flag: T");

REGISTER_OP("BackprojectGrad")
    .Attr("T: {float, double}")
    .Attr("grid_size: int")
    .Attr("kernel_size: int")
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

  // bottom_data: (batch_size, height, width, channels)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();

    const Tensor& bottom_label = context->input(1);
    auto bottom_label_flat = bottom_label.flat<T>();

    const Tensor& bottom_depth = context->input(2);
    auto im_depth = bottom_depth.flat<T>();

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Tensor& bottom_meta_data = context->input(3);
    auto meta_data = bottom_meta_data.flat<T>();

    const Tensor& bottom_label_3d = context->input(4);
    auto bottom_label_3d_flat = bottom_label_3d.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label_3d.dims() == 5,
                errors::InvalidArgument("label 3D must be 5-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_classes = bottom_label.dim_size(3);
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

    // top flag
    TensorShape output_shape_flag;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_flag);

    Tensor* top_flag_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_flag, &top_flag_tensor));
    auto top_flag = top_flag_tensor->template flat<T>();

    // top label
    dims[4] = num_classes;
    TensorShape output_shape_label;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_label);

    Tensor* top_label_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_label, &top_label_tensor));
    auto top_label = top_label_tensor->template flat<T>();

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
            T X = d * meta_data(index_meta_data + 42) + meta_data(index_meta_data + 45);
            T Y = h * meta_data(index_meta_data + 43) + meta_data(index_meta_data + 46);
            T Z = w * meta_data(index_meta_data + 44) + meta_data(index_meta_data + 47);

            // apply pose_world2live
            T X1 = meta_data(index_meta_data + 18) * X + meta_data(index_meta_data + 19) * Y + meta_data(index_meta_data + 20) * Z + meta_data(index_meta_data + 21);
            T Y1 = meta_data(index_meta_data + 22) * X + meta_data(index_meta_data + 23) * Y + meta_data(index_meta_data + 24) * Z + meta_data(index_meta_data + 25);
            T Z1 = meta_data(index_meta_data + 26) * X + meta_data(index_meta_data + 27) * Y + meta_data(index_meta_data + 28) * Z + meta_data(index_meta_data + 29);

            // apply the intrinsic matrix
            T x1 = meta_data(index_meta_data + 0) * X1 + meta_data(index_meta_data + 1) * Y1 + meta_data(index_meta_data + 2) * Z1;
            T x2 = meta_data(index_meta_data + 3) * X1 + meta_data(index_meta_data + 4) * Y1 + meta_data(index_meta_data + 5) * Z1;
            T x3 = meta_data(index_meta_data + 6) * X1 + meta_data(index_meta_data + 7) * Y1 + meta_data(index_meta_data + 8) * Z1;
            int px = round(x1 / x3);
            int py = round(x2 / x3);

            // initialization
            for(int c = 0; c < num_channels; c++)
              top_data((index_batch + index_depth + index_height + w) * num_channels + c) = 0;
            for(int c = 0; c < num_classes; c++)
              top_label((index_batch + index_depth + index_height + w) * num_classes + c) = 0;

            // check a neighborhood around (px, py)
            int count = 0;
            for (int x = px - kernel_size_; x <= px + kernel_size_; x++)
            {
              for (int y = py - kernel_size_; y <= py + kernel_size_; y++)
              {
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                  int index_pixel = n * height * width + y * width + x;
                  T depth = im_depth(index_pixel);
                  // distance of this voxel to camera center
                  T dvoxel = Z1;
                  // check if the voxel is on the surface
                  if (fabs(depth - dvoxel) < threshold_)
                  {
                    count++;
                    // data
                    for(int c = 0; c < num_channels; c++)
                      top_data((index_batch + index_depth + index_height + w) * num_channels + c) += bottom_data_flat(index_pixel * num_channels + c);
                    // label
                    for(int c = 0; c < num_classes; c++)
                      top_label((index_batch + index_depth + index_height + w) * num_classes + c) += bottom_label_flat(index_pixel * num_classes + c);
                  }
                }
              }
            }
            if (count == 0)
            {
              // flag
              for (int c = 0; c < num_channels; c++)
                top_flag((index_batch + index_depth + index_height + w) * num_channels + c) = 0;
              // label
              for(int c = 0; c < num_classes; c++)
                top_label((index_batch + index_depth + index_height + w) * num_classes + c) = bottom_label_3d_flat((index_batch + index_depth + index_height + w) * num_classes + c);
            }
            else
            {
              // data and flag
              for (int c = 0; c < num_channels; c++)
              {
                top_data((index_batch + index_depth + index_height + w) * num_channels + c) /= count;
                top_flag((index_batch + index_depth + index_height + w) * num_channels + c) = 1;
              }
              // label
              for(int c = 0; c < num_classes; c++)
                top_label((index_batch + index_depth + index_height + w) * num_classes + c) /= count;
            }
            // end checking neighborhood
          }
        }
      }
      index_meta_data += num_meta_data;
    }
  }
 private:
  int grid_size_;
  int kernel_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<float>("T"), BackprojectOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Backproject").Device(DEVICE_CPU).TypeConstraint<double>("T"), BackprojectOp<CPUDevice, double>);

bool BackprojectForwardLaucher(
    const float* bottom_data, const float* bottom_label,
    const float* bottom_depth, const float* bottom_meta_data, const float* bottom_label_3d,
    const int batch_size, const int height, const int width, const int channels, const int num_classes, const int num_meta_data,
    const int grid_size, const int kernel_size, const float threshold,
    float* top_data, float* top_label, float* top_flag, const Eigen::GpuDevice& d);

static void BackprojectingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_label,
    const Tensor* bottom_depth, const Tensor* bottom_meta_data, const Tensor* bottom_label_3d,
    const int batch_size, const int height, const int width, const int channels, const int num_classes, const int num_meta_data, 
    const int grid_size, const int kernel_size, const float threshold, 
    const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_label, const TensorShape& tensor_output_shape_flag) 
{
  Tensor* top_data = nullptr;
  Tensor* top_label = nullptr;
  Tensor* top_flag = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_label, &top_label));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape_flag, &top_flag));

  if (!context->status().ok()) {
    return;
  }

  BackprojectForwardLaucher(
    bottom_data->flat<float>().data(), bottom_label->flat<float>().data(),
    bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(), bottom_label_3d->flat<float>().data(),
    batch_size, height, width, channels, num_classes, num_meta_data, grid_size, kernel_size, threshold,
    top_data->flat<float>().data(), top_label->flat<float>().data(), top_flag->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
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
    const Tensor& bottom_label = context->input(1);
    const Tensor& bottom_depth = context->input(2);
    const Tensor& bottom_meta_data = context->input(3);
    const Tensor& bottom_label_3d = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label_3d.dims() == 5,
                errors::InvalidArgument("label 3D must be 5-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_classes = bottom_label.dim_size(3);
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

    // top flag
    TensorShape output_shape_flag;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_flag);

    // top label
    dims[4] = num_classes;
    TensorShape output_shape_label;
    TensorShapeUtils::MakeShape(dims, 5, &output_shape_label);

    BackprojectingKernel(context, &bottom_data, &bottom_label, &bottom_depth, &bottom_meta_data, &bottom_label_3d, batch_size, height,
      width, num_channels, num_classes, num_meta_data, grid_size_, kernel_size_, threshold_, output_shape, output_shape_label, output_shape_flag);
  }
 private:
  int grid_size_;
  int kernel_size_;
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
  int kernel_size_;
  float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("BackprojectGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), BackprojectGradOp<Eigen::GpuDevice, float>);
