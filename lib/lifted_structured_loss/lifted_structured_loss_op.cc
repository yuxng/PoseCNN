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

REGISTER_OP("Liftedstruct")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Attr("budget: int")
    .Input("bottom_data: T")
    .Input("bottom_label: T")
    .Output("loss: T")
    .Output("bottom_diff: T");

REGISTER_OP("LiftedstructGrad")
    .Attr("T: {float, double}")
    .Attr("margin: float")
    .Attr("budget: int")
    .Input("bottom_diff: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class LiftedstructOp : public OpKernel {
 public:
  explicit LiftedstructOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
    // Get the budget
    OP_REQUIRES_OK(context,
                   context->GetAttr("budget", &budget_));
    // Check that budget is positive
    OP_REQUIRES(context, budget_ >= 0,
                errors::InvalidArgument("Need budget >= 0, got ", budget_));
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

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

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

    // sample pixels to define the loss
    // compute label indexes
    std::vector< std::vector<int> > label_indexes(num_classes);
    for (int n = 0; n < batch_size; n++)
    {
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w++)
        {
          int index = n * height * width + h * width + w;
          for (int c = 0; c < num_classes; c++)
          {
            if(labels[index * num_classes + c] > 0)
            {
              label_indexes[c].push_back(index);
              break;
            }
          }
        } 
      }
    }

    // number of classes in the batch
    int count_classes = 0;
    for (int i = 0; i < num_classes; i++)
    {
      if (label_indexes[i].size() > 0)
      {
        count_classes++;
      }
    }

    // decide how many pixels to sample for each class
    int num_pixels_per_class = budget_ / count_classes;

    // sampling
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> pixel_indexes;
    std::vector<int> pixel_labels;
    for (int i = 0; i < num_classes; i++)
    {
      // shuffle the indexes
      std::random_shuffle ( label_indexes[i].begin(), label_indexes[i].end() );
      for (int j = 0; j < label_indexes[i].size() && j < num_pixels_per_class; j++)
      {
        pixel_indexes.push_back(label_indexes[i][j]);
        pixel_labels.push_back(i);
      }
    }

    // construct a eigen matrix of the sampled pixels and its difference
    int num_pixels = pixel_indexes.size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mdata(num_pixels, num_channels);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mdiff(num_pixels, num_channels);
    for(int i = 0; i < num_pixels; i++)
    {
      int index = pixel_indexes[i];
      for(int j = 0; j < num_channels; j++)
      {
        mdata(i, j) = data[index * num_channels + j];
        mdiff(i, j) = 0;
      }
    }

    // compute the distance matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dist_sq(num_pixels, 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ones(num_pixels, 1);
    for(int i = 0; i < num_pixels; i++)
    {
      dist_sq(i, 0) = mdata.row(i).dot(mdata.row(i));
      ones(i, 0) = 1;
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> D(num_pixels, num_pixels);
    D = dist_sq * ones.transpose() + ones * dist_sq.transpose() - 2 * mdata * mdata.transpose();

    // construct pairwise label matrix
    std::vector<std::vector<bool> > label_mat(num_pixels, std::vector<bool>(num_pixels, false));
    for (int i = 0; i < num_pixels; i++)
    {
      for (int j = 0; j < num_pixels; j++)
        label_mat[i][j] = (pixel_labels[i] == pixel_labels[j]);
    }

    T loss = 0;
    T num_constraints = 0;

    // loop upper triangular matrix and look for positive anchors
    for (int i = 0; i < num_pixels; i++)
    {
      for (int j = i + 1; j < num_pixels; j++)
      {
        // found a positive pair @ anchor (i, j)
        if (label_mat[i][j] && D(i, j) > 0)
        {
          T dist_pos = sqrt(D(i, j));

          // 1.count the number of negatives for this positive
          int num_negatives = 0;
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[i][k] && D(i, k) > 0)
              num_negatives++;
          }
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[j][k] && D(j, k) > 0)
              num_negatives++;
          }

          // 2. compute loss augmented inference
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> loss_aug_inference(num_negatives, 1);
          int neg_idx = 0;

          // mine negative (anchor i, neg k)
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[i][k] && D(i, k) > 0)
            {
              loss_aug_inference(neg_idx, 0) = margin_ - sqrt(D(i, k));
              neg_idx++;
            }
          }

          // mine negative (anchor j, neg k)
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[j][k] && D(j, k) > 0)
            {
              loss_aug_inference(neg_idx, 0) = margin_ - sqrt(D(j, k));
              neg_idx++;
            }
          }

          // compute softmax of loss aug inference vector;
          T max_elem = loss_aug_inference.maxCoeff();
          loss_aug_inference = exp(loss_aug_inference.array() - max_elem);
          T sum_exp = loss_aug_inference.sum();
          T soft_maximum = log(sum_exp) + max_elem;

          // hinge the soft_maximum - S_ij (positive pair similarity)
          T this_loss = std::max(soft_maximum + dist_pos, T(0.0));

          // squared hinge
          loss += this_loss * this_loss; 
          num_constraints += 1;

          // 3. compute gradients

          // update from positive distance dJ_dD_{ij}; update x_i, x_j
          T scaler = 2.0 * this_loss / dist_pos;

          // update x_i
          mdiff.row(i) += scaler * (mdata.row(i) - mdata.row(j));

          // update x_j
          mdiff.row(j) += -scaler * (mdata.row(i) - mdata.row(j));

          // update from negative distance dJ_dD_{ik}; update x_i, x_k
          neg_idx = 0;
          T dJ_dDik = 0;
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[i][k] && D(i, k) > 0)
            {
              dJ_dDik = 2.0 * this_loss * (-1.0) * loss_aug_inference(neg_idx, 0) / sum_exp;
              neg_idx++;

              scaler = dJ_dDik / sqrt(D(i, k));

              // update x_i
              mdiff.row(i) += scaler * (mdata.row(i) - mdata.row(k));

              // update x_k
              mdiff.row(k) += -scaler * (mdata.row(i) - mdata.row(k));
            }
          }

          // update from negative distance dJ_dD_{jk}; update x_j, x_k
          T dJ_dDjk = 0;
          for (int k = 0; k < num_pixels; k++)
          {
            if (!label_mat[j][k] && D(j, k) > 0)
            {
              dJ_dDjk = 2.0 * this_loss * (-1.0) * loss_aug_inference(neg_idx, 0) / sum_exp;
              neg_idx++;

              scaler = dJ_dDjk / sqrt(D(j, k));

              // update x_j
              mdiff.row(j) += scaler * (mdata.row(j) - mdata.row(k));

              // update x_k
              mdiff.row(k) += -scaler * (mdata.row(j) - mdata.row(k));
            }
          }
        } // close this postive pair
      }
    }
    loss = loss / num_constraints / T(2.0);
    top_data(0) = loss;

    // construct the gradient
    for (int i = 0; i < num_pixels; i++)
    {
      int index = pixel_indexes[i];
      for (int c = 0; c < num_channels; c++)
      {
        bottom_diff[index * num_channels + c] = mdiff(i, c) / num_constraints / T(2.0);
      }
    }
  }
 private:
  float margin_;
  int budget_;
};

REGISTER_KERNEL_BUILDER(Name("Liftedstruct").Device(DEVICE_CPU).TypeConstraint<float>("T"), LiftedstructOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Liftedstruct").Device(DEVICE_CPU).TypeConstraint<double>("T"), LiftedstructOp<CPUDevice, double>);


// GPU implementation for forward pass
bool LiftedstructForwardLaucher(
    const float* bottom_data, const float* bottom_label,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, const int budget, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);

static void LiftedstructKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_label,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, const int budget, const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff) 
{
  Tensor* top_data = nullptr;
  Tensor* bottom_diff = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_diff));

  if (!context->status().ok()) {
    return;
  }

  LiftedstructForwardLaucher(
    bottom_data->flat<float>().data(), bottom_label->flat<float>().data(),
    batch_size, height, width, channels, num_classes, margin, budget,
    top_data->flat<float>().data(), bottom_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class LiftedstructOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit LiftedstructOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
    // Get the budget
    OP_REQUIRES_OK(context,
                   context->GetAttr("budget", &budget_));
    // Check that budget is positive
    OP_REQUIRES(context, budget_ >= 0,
                errors::InvalidArgument("Need budget >= 0, got ", budget_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_label = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_label.dims() == 4,
                errors::InvalidArgument("label must be 4-dimensional"));

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

    LiftedstructKernel(context, &bottom_data, &bottom_label, batch_size, height,
      width, num_channels, num_classes, margin_, budget_, output_shape, output_shape_diff);
  }
 private:
  float margin_;
  int budget_;
};

REGISTER_KERNEL_BUILDER(Name("Liftedstruct").Device(DEVICE_GPU).TypeConstraint<float>("T"), LiftedstructOp<Eigen::GpuDevice, float>);


// compute gradient
template <class Device, class T>
class LiftedstructGradOp : public OpKernel {
 public:
  explicit LiftedstructGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
    // Get the budget
    OP_REQUIRES_OK(context,
                   context->GetAttr("budget", &budget_));
    // Check that budget is positive
    OP_REQUIRES(context, budget_ >= 0,
                errors::InvalidArgument("Need budget >= 0, got ", budget_));
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
  int budget_;
};

REGISTER_KERNEL_BUILDER(Name("LiftedstructGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), LiftedstructGradOp<CPUDevice, float>);


bool LiftedstructBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d);

static void LiftedstructGradKernel(
    OpKernelContext* context, const Tensor* bottom_diff, const Tensor* out_backprop,
    const int batch_size, const int height, const int width, const int channels,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  LiftedstructBackwardLaucher(
    out_backprop->flat<float>().data(), bottom_diff->flat<float>().data(),
    batch_size, height, width, channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class LiftedstructGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit LiftedstructGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the margin
    OP_REQUIRES_OK(context,
                   context->GetAttr("margin", &margin_));
    // Check that margin is positive
    OP_REQUIRES(context, margin_ >= 0,
                errors::InvalidArgument("Need margin >= 0, got ", margin_));
    // Get the budget
    OP_REQUIRES_OK(context,
                   context->GetAttr("budget", &budget_));
    // Check that budget is positive
    OP_REQUIRES(context, budget_ >= 0,
                errors::InvalidArgument("Need budget >= 0, got ", budget_));
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
    LiftedstructGradKernel(
      context, &bottom_diff, &out_backprop, batch_size, height, width, num_channels, output_shape);
  }
 private:
  float margin_;
  int budget_;
};

REGISTER_KERNEL_BUILDER(Name("LiftedstructGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), LiftedstructGradOp<Eigen::GpuDevice, float>);
