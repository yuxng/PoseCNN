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

// Computing Flow Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Computeflow")
    .Attr("T: {float, double}")
    .Attr("kernel_size: int")
    .Attr("threshold: float")
    .Attr("max_weight: float")
    .Input("bottom_data: T")
    .Input("bottom_weights: T")
    .Input("bottom_points: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Output("top_data: T")
    .Output("top_weights: T")
    .Output("top_points: T");

REGISTER_OP("ComputeflowGrad")
    .Attr("T: {float, double}")
    .Attr("kernel_size: int")
    .Attr("threshold: float")
    .Attr("max_weight: float")
    .Input("bottom_data: T")
    .Input("bottom_weights: T")
    .Input("bottom_points: T")
    .Input("bottom_depth: T")
    .Input("bottom_meta_data: T")
    .Input("top_points: T")
    .Input("grad: T")
    .Input("grad_weights: T")
    .Output("output: T")
    .Output("output_weights: T");

template <typename Device, typename T>
class ComputeFlowOp : public OpKernel {
 public:
  explicit ComputeFlowOp(OpKernelConstruction* context) : OpKernel(context) {
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
    // Get the max weight
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_weight", &max_weight_));
    // Check that max weight is positive
    OP_REQUIRES(context, max_weight_ >= 0,
                errors::InvalidArgument("Need max_weight >= 0, got ", max_weight_));
  }

  // bottom_data: (batch_size, height, width, channels)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();

    const Tensor& bottom_weights = context->input(1);
    auto bottom_weights_flat = bottom_weights.flat<T>();
    
    const Tensor& bottom_points = context->input(2);
    auto im_points = bottom_points.flat<T>();

    const Tensor& bottom_depth = context->input(3);
    auto im_depth = bottom_depth.flat<T>();

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Tensor& bottom_meta_data = context->input(4);
    auto meta_data = bottom_meta_data.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));
    
    OP_REQUIRES(context, bottom_weights.dims() == 4,
                errors::InvalidArgument("weights must be 4-dimensional"));

    OP_REQUIRES(context, bottom_points.dims() == 4,
                errors::InvalidArgument("points must be 4-dimensional"));

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

    // top weights
    Tensor* top_weights_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &top_weights_tensor));
    auto top_weights = top_weights_tensor->template flat<T>();

    // top points
    dims[3] = 3;
    TensorShape output_shape_1;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape_1);
    Tensor* top_points_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_1, &top_points_tensor));
    auto top_points = top_points_tensor->template flat<T>();

    int index_meta_data = 0;    
    for(int n = 0; n < batch_size; n++)
    {
      for(int h = 0; h < height; h++)
      {
        for(int w = 0; w < width; w++)
        {
          int index_pixel = n * height * width + h * width + w;

          // initialization
          for(int c = 0; c < num_channels; c++)
          {
            top_data(index_pixel * num_channels + c) = 0;
            top_weights(index_pixel * num_channels + c) = 1;
          }
          top_points(index_pixel * 3 + 0) = NAN;
          top_points(index_pixel * 3 + 1) = NAN;
          top_points(index_pixel * 3 + 2) = NAN;

          T depth = im_depth(index_pixel);
          if (depth > 0)
          {
            // backproject the pixel into 3D
            // apply the inverse intrinsic matrix
            T RX = meta_data(index_meta_data + 9) * w + meta_data(index_meta_data + 10) * h + meta_data(index_meta_data + 11);
            T RY = meta_data(index_meta_data + 12) * w + meta_data(index_meta_data + 13) * h + meta_data(index_meta_data + 14);
            T RZ = meta_data(index_meta_data + 15) * w + meta_data(index_meta_data + 16) * h + meta_data(index_meta_data + 17);

            // compute the 3D points in camera's coordinate system
            T X = depth * RX;
            T Y = depth * RY;
            T Z = depth * RZ;

            // store the points
            top_points(index_pixel * 3 + 0) = X;
            top_points(index_pixel * 3 + 1) = Y;
            top_points(index_pixel * 3 + 2) = Z;

            // apply pose_live2world
            T X1 = meta_data(index_meta_data + 30) * X + meta_data(index_meta_data + 31) * Y + meta_data(index_meta_data + 32) * Z + meta_data(index_meta_data + 33);
            T Y1 = meta_data(index_meta_data + 34) * X + meta_data(index_meta_data + 35) * Y + meta_data(index_meta_data + 36) * Z + meta_data(index_meta_data + 37);
            T Z1 = meta_data(index_meta_data + 38) * X + meta_data(index_meta_data + 39) * Y + meta_data(index_meta_data + 40) * Z + meta_data(index_meta_data + 41);

            // project the point into world image
            // apply the intrinsic matrix
            T x1 = meta_data(index_meta_data + 0) * X1 + meta_data(index_meta_data + 1) * Y1 + meta_data(index_meta_data + 2) * Z1;
            T x2 = meta_data(index_meta_data + 3) * X1 + meta_data(index_meta_data + 4) * Y1 + meta_data(index_meta_data + 5) * Z1;
            T x3 = meta_data(index_meta_data + 6) * X1 + meta_data(index_meta_data + 7) * Y1 + meta_data(index_meta_data + 8) * Z1;
            int px = round(x1 / x3);
            int py = round(x2 / x3);

            // averaging over a small neighborhood
            int count = 0;
            for (int x = px - kernel_size_; x <= px + kernel_size_; x++)
            {
              for (int y = py - kernel_size_; y <= py + kernel_size_; y++)
              {
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                  int index = n * height * width + y * width + x;
                  T Z_prev = im_points(index * 3 + 2);
                  if (fabs(Z_prev - Z1) < threshold_)
                  {
                    for(int c = 0; c < num_channels; c++)
                    {
                      top_data(index_pixel * num_channels + c) = (bottom_data_flat(index * num_channels + c) + count * top_data(index_pixel * num_channels + c)) / (count + 1);
                      T weight = bottom_weights_flat(index * num_channels + c);
                      if (weight > max_weight_)
                        top_weights(index_pixel * num_channels + c) = (max_weight_ + count * top_weights(index_pixel * num_channels + c)) / (count + 1);
                      else
                        top_weights(index_pixel * num_channels + c) = (weight + count * top_weights(index_pixel * num_channels + c)) / (count + 1);
                    }
                    count++;
                  }
                }
              }
            }
          }
        }
      }
      index_meta_data += num_meta_data;
    }
  }
 private:
  int kernel_size_;
  float threshold_;
  float max_weight_;
};

REGISTER_KERNEL_BUILDER(Name("Computeflow").Device(DEVICE_CPU).TypeConstraint<float>("T"), ComputeFlowOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Computeflow").Device(DEVICE_CPU).TypeConstraint<double>("T"), ComputeFlowOp<CPUDevice, double>);

bool ComputeFlowForwardLaucher(
    const float* bottom_data, const float* bottom_weights, const float* bottom_points,
    const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data,
    const int kernel_size, const float threshold, const float max_weight,
    float* top_data, float* top_weights, float* top_points, const Eigen::GpuDevice& d);

static void ComputingFlowKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_weights, const Tensor* bottom_points,
    const Tensor* bottom_depth, const Tensor* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, 
    const int kernel_size, const float threshold, const float max_weight,
    const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_points) 
{
  Tensor* top_data = nullptr;
  Tensor* top_weights = nullptr;
  Tensor* top_points = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &top_weights));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape_points, &top_points));

  if (!context->status().ok()) {
    return;
  }

  ComputeFlowForwardLaucher(
    bottom_data->flat<float>().data(), bottom_weights->flat<float>().data(), bottom_points->flat<float>().data(),
    bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, kernel_size, threshold, max_weight,
    top_data->flat<float>().data(), top_weights->flat<float>().data(), top_points->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class ComputeFlowOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit ComputeFlowOp(OpKernelConstruction* context) : OpKernel(context) {
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
    // Get the max weight
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_weight", &max_weight_));
    // Check that max weight is positive
    OP_REQUIRES(context, max_weight_ >= 0,
                errors::InvalidArgument("Need max_weight >= 0, got ", max_weight_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_weights = context->input(1);
    const Tensor& bottom_points = context->input(2);
    const Tensor& bottom_depth = context->input(3);
    const Tensor& bottom_meta_data = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_weights.dims() == 4,
                errors::InvalidArgument("weights must be 4-dimensional"));
    
    OP_REQUIRES(context, bottom_points.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

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
    // top_data and top_weights
    int dims[4];
    dims[0] = batch_size;
    dims[1] = height;
    dims[2] = width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // top points
    dims[3] = 3;
    TensorShape output_shape_points;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape_points);
    
    ComputingFlowKernel(context, &bottom_data, &bottom_weights, &bottom_points, &bottom_depth, &bottom_meta_data, batch_size, height,
      width, num_channels, num_meta_data, kernel_size_, threshold_, max_weight_, output_shape, output_shape_points);
  }
 private:
  int kernel_size_;
  float threshold_;
  float max_weight_;
};

REGISTER_KERNEL_BUILDER(Name("Computeflow").Device(DEVICE_GPU).TypeConstraint<float>("T"), ComputeFlowOp<Eigen::GpuDevice, float>);


bool ComputeFlowBackwardLaucher(const float* top_diff, const float* top_diff_weights, const float* bottom_weights, 
    const float* bottom_points, const float* bottom_depth, const float* bottom_meta_data, const float* top_points, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int kernel_size, const float threshold, const float max_weight,
    float* bottom_diff, float* bottom_diff_weights, const Eigen::GpuDevice& d);

static void ComputingFlowGradKernel(
    OpKernelContext* context, const Tensor* bottom_weights, const Tensor* bottom_points, const Tensor* bottom_depth, const Tensor* bottom_meta_data,
    const Tensor* top_points, const Tensor* out_backprop, const Tensor* out_backprop_weights,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, const int kernel_size, const float threshold, const float max_weight,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  Tensor* output_weights = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &output_weights));

  if (!context->status().ok()) {
    return;
  }

  ComputeFlowBackwardLaucher(
    out_backprop->flat<float>().data(), out_backprop_weights->flat<float>().data(), bottom_weights->flat<float>().data(), bottom_points->flat<float>().data(), 
    bottom_depth->flat<float>().data(), bottom_meta_data->flat<float>().data(), top_points->flat<float>().data(),
    batch_size, height, width, channels, num_meta_data, kernel_size, threshold, max_weight, output->flat<float>().data(), 
    output_weights->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class ComputeFlowGradOp : public OpKernel {
 public:
  explicit ComputeFlowGradOp(OpKernelConstruction* context) : OpKernel(context) {
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
    // Get the max weight
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_weight", &max_weight_));
    // Check that max weight is positive
    OP_REQUIRES(context, max_weight_ >= 0,
                errors::InvalidArgument("Need max_weight >= 0, got ", max_weight_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_weights = context->input(1);
    const Tensor& bottom_points = context->input(2);
    const Tensor& bottom_depth = context->input(3);
    const Tensor& bottom_meta_data = context->input(4);
    const Tensor& top_points = context->input(5);
    const Tensor& out_backprop = context->input(6);
    const Tensor& out_backprop_weights = context->input(7);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_weights.dims() == 4,
                errors::InvalidArgument("weights must be 4-dimensional"));

    OP_REQUIRES(context, bottom_points.dims() == 4,
                errors::InvalidArgument("bottom points must be 4-dimensional"));

    OP_REQUIRES(context, bottom_depth.dims() == 4,
                errors::InvalidArgument("depth must be 4-dimensional"));

    OP_REQUIRES(context, bottom_meta_data.dims() == 4,
                errors::InvalidArgument("meta data must be 4-dimensional"));

    OP_REQUIRES(context, top_points.dims() == 4,
                errors::InvalidArgument("top points must be 4-dimensional"));

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

    ComputingFlowGradKernel(
      context, &bottom_weights, &bottom_points, &bottom_depth, &bottom_meta_data, &top_points, &out_backprop, &out_backprop_weights,
      batch_size, height, width, num_channels, num_meta_data, kernel_size_, threshold_, max_weight_, output_shape);

  }
 private:
  int kernel_size_;
  float threshold_;
  float max_weight_;
};

REGISTER_KERNEL_BUILDER(Name("ComputeflowGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), ComputeFlowGradOp<Eigen::GpuDevice, float>);
