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

// Loss Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 
#include <vector>
#include <ctime>
#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#define POSE_CHANNELS 4

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Averagedistance")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Input("bottom_prediction: T")
    .Input("bottom_target: T")
    .Input("bottom_weight: T")
    .Input("bottom_point: T")
    .Input("bottom_symmetry: T")
    .Output("loss: T")
    .Output("bottom_diff: T");

REGISTER_OP("AveragedistanceGrad")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Input("bottom_diff: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class AveragedistanceOp : public OpKernel {
 public:
  explicit AveragedistanceOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
  }

  // bottom_prediction: (batch_size, 4 * num_classes)
  // bottom_point: (num_classes, num_points, 3)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prediction = context->input(0);
    const T* prediction = bottom_prediction.flat<T>().data();

    const Tensor& bottom_target = context->input(1);
    const T* target = bottom_target.flat<T>().data();

    const Tensor& bottom_weight = context->input(2);
    const T* weight = bottom_weight.flat<T>().data();

    const Tensor& bottom_point = context->input(3);
    const T* point = bottom_point.flat<T>().data();

    const Tensor& bottom_symmetry = context->input(4);
    const T* symmetry = bottom_symmetry.flat<T>().data();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_prediction.dims() == 2,
                errors::InvalidArgument("prediction must be 2-dimensional"));

    OP_REQUIRES(context, bottom_target.dims() == 2,
                errors::InvalidArgument("target must be 2-dimensional"));

    OP_REQUIRES(context, bottom_weight.dims() == 2,
                errors::InvalidArgument("weight must be 2-dimensional"));

    OP_REQUIRES(context, bottom_point.dims() == 3,
                errors::InvalidArgument("point must be 3-dimensional"));

    OP_REQUIRES(context, bottom_symmetry.dims() == 1,
                errors::InvalidArgument("symmetry must be 1-dimensional"));

    // batch size
    int batch_size = bottom_prediction.dim_size(0);
    int num_classes = bottom_point.dim_size(0);
    int num_points = bottom_point.dim_size(1);

    // Create output loss tensor
    int dim = 1;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    // bottom diff
    TensorShape output_shape_diff = bottom_prediction.shape();
    Tensor* bottom_diff_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_diff, &bottom_diff_tensor));
    T* bottom_diff = bottom_diff_tensor->template flat<T>().data();
    memset(bottom_diff, 0, batch_size * POSE_CHANNELS * num_classes *sizeof(T));

    T loss = 0;
    // for each object
    for (int n = 0; n < batch_size; n++)
    {
      // find the class label and pose of this object
      int index_cls = -1;
      Eigen::Quaternionf pose_gt;
      Eigen::Quaternionf pose_u;
      for (int i = 0; i < POSE_CHANNELS * num_classes; i += POSE_CHANNELS)
      {
        int index = n * POSE_CHANNELS * num_classes + i;
        if (weight[index] > 0)
        {
          index_cls = i / POSE_CHANNELS;

          pose_gt.w() = target[index + 0];
          pose_gt.x() = target[index + 1];
          pose_gt.y() = target[index + 2];
          pose_gt.z() = target[index + 3];

          pose_u.w() = prediction[index + 0];
          pose_u.x() = prediction[index + 1];
          pose_u.y() = prediction[index + 2];
          pose_u.z() = prediction[index + 3];
          break;
        }
      }
      if (index_cls == -1)
        continue;

      // rotation matrix
      Eigen::Matrix3f Rgt = pose_gt.toRotationMatrix();
      Eigen::Matrix3f Ru = pose_u.toRotationMatrix();

      // derivatives of Ru to quaternion
      Eigen::Matrix3f Ru_w;
      Ru_w << 2 * pose_u.w(), -2 * pose_u.z(), 2 * pose_u.y(),
              2 * pose_u.z(), 2 * pose_u.w(), -2 * pose_u.x(),
              -2 * pose_u.y(), 2 * pose_u.x(), 2 * pose_u.w();

      Eigen::Matrix3f Ru_x;
      Ru_x << 2 * pose_u.x(), 2 * pose_u.y(), 2 * pose_u.z(),
              2 * pose_u.y(), -2 * pose_u.x(), -2 * pose_u.w(),
              2 * pose_u.z(), 2 * pose_u.w(), -2 * pose_u.x();

      Eigen::Matrix3f Ru_y;
      Ru_y << -2 * pose_u.y(), 2 * pose_u.x(), 2 * pose_u.w(),
              2 * pose_u.x(), 2 * pose_u.y(), 2 * pose_u.z(),
              -2 * pose_u.w(), 2 * pose_u.z(), -2 * pose_u.y();

      Eigen::Matrix3f Ru_z;
      Ru_z << -2 * pose_u.z(), -2 * pose_u.w(), 2 * pose_u.x(),
              2 * pose_u.w(), -2 * pose_u.z(), 2 * pose_u.y(),
              2 * pose_u.x(), 2 * pose_u.y(), 2 * pose_u.z();

      // for each point
      for (int i = 0; i < num_points; i++)
      {
        int index = index_cls * num_points * 3 + i * 3;
        Eigen::Vector3f x3d(point[index], point[index + 1], point[index + 2]);
        Eigen::Vector3f diff = Ru * x3d - Rgt * x3d;
        T distance = diff.dot(diff);
        if (distance < margin_)
          continue;
        loss += (distance - margin_) / 2.0;

        // compute the gradient from this point
        Eigen::Matrix3f f0;
        f0 << x3d[0], x3d[1], x3d[2],
              0, 0, 0,
              0, 0, 0;

        Eigen::Matrix3f f1;
        f1 << 0, 0, 0,
              x3d[0], x3d[1], x3d[2],
              0, 0, 0;

        Eigen::Matrix3f f2;
        f2 << 0, 0, 0,
              0, 0, 0,
              x3d[0], x3d[1], x3d[2];

        Eigen::Matrix3f f = diff[0] * f0 + diff[1] * f1 + diff[2] * f2;

        int index_diff = n * POSE_CHANNELS * num_classes + POSE_CHANNELS * index_cls;
        bottom_diff[index_diff + 0] += f.cwiseProduct(Ru_w).sum() / (batch_size * num_points);
        bottom_diff[index_diff + 1] += f.cwiseProduct(Ru_x).sum() / (batch_size * num_points);
        bottom_diff[index_diff + 2] += f.cwiseProduct(Ru_y).sum() / (batch_size * num_points);
        bottom_diff[index_diff + 3] += f.cwiseProduct(Ru_z).sum() / (batch_size * num_points);
      }
    }
    loss /= batch_size * num_points;
    top_data(0) = loss;
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("Averagedistance").Device(DEVICE_CPU).TypeConstraint<float>("T"), AveragedistanceOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Averagedistance").Device(DEVICE_CPU).TypeConstraint<double>("T"), AveragedistanceOp<CPUDevice, double>);

// GPU implementation for forward pass
bool AveragedistanceForwardLaucher(OpKernelContext* context,
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const float* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);

static void AveragedistanceKernel(
    OpKernelContext* context, const Tensor* bottom_prediction, const Tensor* bottom_target, const Tensor* bottom_weight,
    const Tensor* bottom_point, const Tensor* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff) 
{
  Tensor* top_data = nullptr;
  Tensor* bottom_diff = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_diff));

  if (!context->status().ok()) {
    return;
  }

   AveragedistanceForwardLaucher(context,
    bottom_prediction->flat<float>().data(), bottom_target->flat<float>().data(), bottom_weight->flat<float>().data(),
    bottom_point->flat<float>().data(), bottom_symmetry->flat<float>().data(), batch_size, num_classes, num_points, margin,
    top_data->flat<float>().data(), bottom_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class AveragedistanceOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit AveragedistanceOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prediction = context->input(0);
    const Tensor& bottom_target = context->input(1);
    const Tensor& bottom_weight = context->input(2);
    const Tensor& bottom_point = context->input(3);
    const Tensor& bottom_symmetry = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_prediction.dims() == 2,
                errors::InvalidArgument("prediction must be 2-dimensional"));

    OP_REQUIRES(context, bottom_target.dims() == 2,
                errors::InvalidArgument("target must be 2-dimensional"));

    OP_REQUIRES(context, bottom_weight.dims() == 2,
                errors::InvalidArgument("weight must be 2-dimensional"));

    OP_REQUIRES(context, bottom_point.dims() == 3,
                errors::InvalidArgument("point must be 3-dimensional"));

    OP_REQUIRES(context, bottom_symmetry.dims() == 1,
                errors::InvalidArgument("symmetry must be 1-dimensional"));

    // batch size
    int batch_size = bottom_prediction.dim_size(0);
    int num_classes = bottom_point.dim_size(0);
    int num_points = bottom_point.dim_size(1);

    // Create output tensors
    // loss
    int dim = 1;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    // bottom diff
    TensorShape output_shape_diff = bottom_prediction.shape();

    AveragedistanceKernel(context, &bottom_prediction, &bottom_target, &bottom_weight, &bottom_point, &bottom_symmetry, batch_size, num_classes,
      num_points, margin_, output_shape, output_shape_diff);
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("Averagedistance").Device(DEVICE_GPU).TypeConstraint<float>("T"), AveragedistanceOp<Eigen::GpuDevice, float>);

// compute gradient
template <class Device, class T>
class AveragedistanceGradOp : public OpKernel {
 public:
  explicit AveragedistanceGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
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
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("AveragedistanceGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), AveragedistanceGradOp<CPUDevice, float>);

bool AveragedistanceBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int channels, float* output, const Eigen::GpuDevice& d);

static void AveragedistanceGradKernel(
    OpKernelContext* context, const Tensor* bottom_diff, const Tensor* out_backprop,
    const int batch_size, const int channels,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  AveragedistanceBackwardLaucher(
    out_backprop->flat<float>().data(), bottom_diff->flat<float>().data(),
    batch_size, channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class AveragedistanceGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit AveragedistanceGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
  }

  void Compute(OpKernelContext* context) override 
  {
    const Tensor& bottom_diff = context->input(0);
    const Tensor& out_backprop = context->input(1);

    // data should have 2 dimensions.
    OP_REQUIRES(context, bottom_diff.dims() == 2,
                errors::InvalidArgument("bottom diff must be 2-dimensional"));

    // batch size
    int batch_size = bottom_diff.dim_size(0);
    // number of channels
    int num_channels = bottom_diff.dim_size(1);

    // construct the output shape
    TensorShape output_shape = bottom_diff.shape();

    // run the kernel
    AveragedistanceGradKernel(
      context, &bottom_diff, &out_backprop, batch_size, num_channels, output_shape);
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("AveragedistanceGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), AveragedistanceGradOp<Eigen::GpuDevice, float>);
