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

// Lifted Structured Loss Op

#include <stdio.h>
#include <cfloat>
#include <math.h> 
#include <vector>
#include <ctime>
#include <cstdlib>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Triplet")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Input("bottom_data: T")
    .Input("bottom_label: T")
    .Input("bottom_prediction: int32")
    .Output("loss: T")
    .Output("bottom_diff: T");

REGISTER_OP("TripletGrad")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Input("bottom_diff: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class TripletOp : public OpKernel {
 public:
  explicit TripletOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
  }

  // bottom_data: (batch_size, height, width, channels)
  // bottom_label: (batch_size, height, width, num_classes)
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const T* data = bottom_data.flat<T>().data();

    const Tensor& bottom_label = context->input(1);
    const T* labels = bottom_label.flat<T>().data();

    const Tensor& bottom_prediction = context->input(2);
    const int* predictions = bottom_prediction.flat<int>().data();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

    OP_REQUIRES(context, bottom_prediction.dims() == 3,
                errors::InvalidArgument("prediction must be 3-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_classes = bottom_label.dim_size(3);

    // Create output loss tensor
    int dim = 1;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    Tensor* top_data_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
    auto top_data = top_data_tensor->template flat<T>();

    // bottom diff
    TensorShape output_shape_diff = bottom_data.shape();
    Tensor* bottom_diff_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_diff, &bottom_diff_tensor));
    T* bottom_diff = bottom_diff_tensor->template flat<T>().data();
    memset(bottom_diff, 0, batch_size * height * width * num_channels *sizeof(T));

    // sample triplets to define the loss
    // compute label indexes
    std::vector< std::vector<int> > label_indexes(num_classes);
    std::vector< std::vector<int> > label_indexes_correct(num_classes);
    for (int n = 0; n < batch_size; n++)
    {
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w++)
        {
          int index = n * height * width + h * width + w;
          int cls;
          for (int c = 0; c < num_classes; c++)
          {
            if(labels[index * num_classes + c] > 0)
            {
              label_indexes[c].push_back(index);
              cls = c;
              break;
            }
          }
          if (predictions[index] == cls)
            label_indexes_correct[cls].push_back(index);
        } 
      }
    }

    // classes in the batch
    std::vector<int> class_indexes;
    for (int i = 0; i < num_classes; i++)
    {
      if (label_indexes[i].size() > 0)
        class_indexes.push_back(i);
    }

    // sampling
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> triplets(batch_size * height * width * 3);
    for (int n = 0; n < batch_size; n++)
    {
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w++)
        {
          // anchor
          int index = n * height * width + h * width + w;
          int cls;
          for (int c = 0; c < num_classes; c++)
          {
            if(labels[index * num_classes + c] > 0)
            {
              cls = c;
              break;
            }
          }

          // sample a positive pixel
          int num = label_indexes_correct[cls].size();
          int index_p;
          if (num > 0 && rand() % 2 == 0)
          {
            if (num == 1)
              index_p = label_indexes_correct[cls][0];
            else
            {
              while(1)
              {
                index_p = label_indexes_correct[cls][rand() % num];
                if (index_p != index)
                  break;
              }
            }
          }
          else
          {
            num = label_indexes[cls].size();
            if (num == 1)
              index_p = index;
            else
            {
              while(1)
              {
                index_p = label_indexes[cls][rand() % num];
                if (index_p != index)
                  break;
              }
            }
          }

          // sample a negative pixel
          int cls_neg;
          // check the predicted label of this pixel for hard negative
          int cls_pred = predictions[index];
          if (cls_pred != cls && label_indexes[cls_pred].size() > 0 && rand() % 2 == 0)
            cls_neg = cls_pred;
          else
          {
            while(1)
            {
              cls_neg = class_indexes[rand() % class_indexes.size()];
              if (cls_neg != cls)
                break;
            }
          }
          int index_n;
          num = label_indexes_correct[cls_neg].size();
          if (num > 0 && rand() % 2 == 0)
            index_n = label_indexes_correct[cls_neg][rand() % num];
          else
          {
            num = label_indexes[cls_neg].size();
            index_n = label_indexes[cls_neg][rand() % num];
          }

          // store the triplet
          triplets[index * 3 + 0] = index;
          triplets[index * 3 + 1] = index_p;
          triplets[index * 3 + 2] = index_n;
        } 
      }
    }

    T loss = 0;
    // for each triplet
    int num_triplets = batch_size * height * width;
    for (int n = 0; n < batch_size; n++)
    {
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w++)
        {
          int index = n * height * width + h * width + w;
          int index_i = triplets[index * 3 + 0];
          int index_j = triplets[index * 3 + 1];
          int index_k = triplets[index * 3 + 2];

          // compute the distances
          T D_ij = 0;
          T D_ik = 0;
          for (int c = 0; c < num_channels; c++)
          {
            D_ij += (data[index_i * num_channels + c] - data[index_j * num_channels + c]) * (data[index_i * num_channels + c] - data[index_j * num_channels + c]);
            D_ik += (data[index_i * num_channels + c] - data[index_k * num_channels + c]) * (data[index_i * num_channels + c] - data[index_k * num_channels + c]);
          }

          // add the loss
          T dis = D_ij - D_ik + margin_;
          loss += std::max(dis, T(0.0));

          // compute gradients
          if (dis > 0)
          {
            for (int c = 0; c < num_channels; c++)
            {
              // update x_i
              bottom_diff[index_i * num_channels + c] += (data[index_k * num_channels + c] - data[index_j * num_channels + c]) / num_triplets;
              // update x_j
              bottom_diff[index_j * num_channels + c] += (data[index_j * num_channels + c] - data[index_i * num_channels + c]) / num_triplets;
              // update x_k
              bottom_diff[index_k * num_channels + c] += (data[index_i * num_channels + c] - data[index_k * num_channels + c]) / num_triplets;
            }
          }
        }
      }
    }
    loss /= num_triplets * 2.0;
    top_data(0) = loss;
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("Triplet").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Triplet").Device(DEVICE_CPU).TypeConstraint<double>("T"), TripletOp<CPUDevice, double>);


// GPU implementation for forward pass
bool TripletForwardLaucher(
    const float* bottom_data, const float* bottom_label, const int* bottom_prediction,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);

static void TripletKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_label, const Tensor* bottom_prediction,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff) 
{
  Tensor* top_data = nullptr;
  Tensor* bottom_diff = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_diff));

  if (!context->status().ok()) {
    return;
  }

  TripletForwardLaucher(
    bottom_data->flat<float>().data(), bottom_label->flat<float>().data(), bottom_prediction->flat<int>().data(),
    batch_size, height, width, channels, num_classes, margin,
    top_data->flat<float>().data(), bottom_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class TripletOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit TripletOp(OpKernelConstruction* context) : OpKernel(context) {
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
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_label = context->input(1);
    const Tensor& bottom_prediction = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

    OP_REQUIRES(context, bottom_prediction.dims() == 3,
                errors::InvalidArgument("prediction must be 3-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // height
    int height = bottom_data.dim_size(1);
    // width
    int width = bottom_data.dim_size(2);
    // number of channels
    int num_channels = bottom_data.dim_size(3);
    int num_classes = bottom_label.dim_size(3);

    // Create output tensors
    // loss
    int dim = 1;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

    // bottom diff
    TensorShape output_shape_diff = bottom_data.shape();

    TripletKernel(context, &bottom_data, &bottom_label, &bottom_prediction, batch_size, height,
      width, num_channels, num_classes, margin_, output_shape, output_shape_diff);
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("Triplet").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletOp<Eigen::GpuDevice, float>);


// compute gradient
template <class Device, class T>
class TripletGradOp : public OpKernel {
 public:
  explicit TripletGradOp(OpKernelConstruction* context) : OpKernel(context) {
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
    OP_REQUIRES(context, bottom_diff.dims() == 4,
                errors::InvalidArgument("bottom diff must be 4-dimensional"));

    // batch size
    int batch_size = bottom_diff.dim_size(0);
    // height
    int height = bottom_diff.dim_size(1);
    // width
    int width = bottom_diff.dim_size(2);
    // number of channels
    int num_channels = bottom_diff.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_diff.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto top_data = output->template flat<T>();

    for (int i = 0; i < batch_size * height * width * num_channels; i++)
      top_data(i) = loss * bottom_diff_flat(i);
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("TripletGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletGradOp<CPUDevice, float>);


bool TripletBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d);

static void TripletGradKernel(
    OpKernelContext* context, const Tensor* bottom_diff, const Tensor* out_backprop,
    const int batch_size, const int height, const int width, const int channels,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  TripletBackwardLaucher(
    out_backprop->flat<float>().data(), bottom_diff->flat<float>().data(),
    batch_size, height, width, channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class TripletGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit TripletGradOp(OpKernelConstruction* context) : OpKernel(context) {
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

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_diff.dims() == 4,
                errors::InvalidArgument("bottom diff must be 4-dimensional"));

    // batch size
    int batch_size = bottom_diff.dim_size(0);
    // height
    int height = bottom_diff.dim_size(1);
    // width
    int width = bottom_diff.dim_size(2);
    // number of channels
    int num_channels = bottom_diff.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_diff.shape();

    // run the kernel
    TripletGradKernel(
      context, &bottom_diff, &out_backprop, batch_size, height, width, num_channels, output_shape);
  }
 private:
  float margin_;
};

REGISTER_KERNEL_BUILDER(Name("TripletGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletGradOp<Eigen::GpuDevice, float>);
