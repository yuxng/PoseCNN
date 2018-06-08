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

// Computing label Op

#include <stdio.h>
#include <cfloat>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <Eigen/Geometry> 
#include "opencv2/opencv.hpp"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#define VERTEX_CHANNELS 3
#define MAX_ROI 128

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Houghvotinggpu")
    .Attr("T: {float, double}")
    .Attr("is_train: int")
    .Attr("threshold_vote: float")
    .Attr("threshold_percentage: float")
    .Attr("skip_pixels: int")
    .Input("bottom_label: int32")
    .Input("bottom_vertex: T")
    .Input("bottom_extents: T")
    .Input("bottom_meta_data: T")
    .Input("bottom_gt: T")
    .Output("top_box: T")
    .Output("top_pose: T")
    .Output("top_target: T")
    .Output("top_weight: T")
    .Output("top_domain: int32");

REGISTER_OP("HoughvotinggpuGrad")
    .Attr("T: {float, double}")
    .Input("bottom_label: int32")
    .Input("bottom_vertex: T")
    .Input("grad: T")
    .Output("output_label: T")
    .Output("output_vertex: T");

int clamp(int val, int min_val, int max_val)
{
  return std::max(min_val, std::min(max_val, val));
}

void getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent);
inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& rvec, const cv::Mat& tvec);
inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2);
inline float angle_distance(cv::Point2f x, cv::Point2f n, cv::Point2f p);

void hough_voting(const int* labelmap, const float* vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds,
  int batch, int height, int width, int num_classes, int is_train,
  float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> >& outputs);

void compute_target_weight(int height, int width, float* target, float* weight, std::vector<std::vector<cv::Point3f>> bb3Ds, 
  const float* poses_gt, int num_gt, int num_classes, float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> > outputs);

inline void compute_width_height(const int* labelmap, const float* vertmap, cv::Point2f center, 
  std::vector<std::vector<cv::Point3f>> bb3Ds, cv::Mat camMat, float inlierThreshold, 
  int height, int width, int channel, int num_classes, int & bb_width, int & bb_height, float & bb_distance);

// cuda functions
void HoughVotingLaucher(OpKernelContext* context,
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
    const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold,
    const int skip_pixels,float* top_box, float* top_pose, float* top_target, float* top_weight, 
    int* top_domain, int* num_rois, const Eigen::GpuDevice& d);

void allocate_outputs(OpKernelContext* context, Tensor* top_box_tensor, Tensor* top_pose_tensor, Tensor* top_target_tensor, Tensor* top_weight_tensor, Tensor* top_domain_tensor, Tensor* top_rois_tensor, int num_classes)
{
  int num = MAX_ROI * 9;
  int dims[2];

  dims[0] = num;
  dims[1] = 7;
  TensorShape output_shape;
  TensorShapeUtils::MakeShape(dims, 2, &output_shape);
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape, top_box_tensor));

  dims[1] = 7;
  TensorShape output_shape_1;
  TensorShapeUtils::MakeShape(dims, 2, &output_shape_1);
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_1, top_pose_tensor));

  dims[1] = 4 * num_classes;
  TensorShape output_shape_2;
  TensorShapeUtils::MakeShape(dims, 2, &output_shape_2);
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_2, top_target_tensor));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_2, top_weight_tensor));

  TensorShape output_shape_3;
  TensorShapeUtils::MakeShape(&num, 1, &output_shape_3);
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_3, top_domain_tensor));

  int len = 1;
  TensorShape output_shape_4;
  TensorShapeUtils::MakeShape(&len, 1, &output_shape_4);
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_4, top_rois_tensor));
}

void reset_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, int num_classes);
void copy_num_rois(int* num_rois, int* num_rois_device);

void copy_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain,
  float* top_box_final, float* top_pose_final, float* top_target_final, float* top_weight_final, int* top_domain_final, int num_classes, int num_rois);

void set_gradients(float* top_label, float* top_vertex, int batch_size, int height, int width, int num_classes);

template <typename Device, typename T>
class HoughvotinggpuOp : public OpKernel {
 public:
  explicit HoughvotinggpuOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_train", &is_train_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, is_train_ >= 0,
                errors::InvalidArgument("Need is_train >= 0, got ",
                                        is_train_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold_vote", &threshold_vote_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold_percentage", &threshold_percentage_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_pixels", &skip_pixels_));

  }

  // bottom_label: (batch_size, height, width)
  // bottom_vertex: (batch_size, height, width, 3 * num_classes)
  // top_box: (num, 7) i.e., batch_index, cls, x1, y1, x2, y2, score
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_label = context->input(0);
    const Tensor& bottom_vertex = context->input(1);
    const Tensor& bottom_extents = context->input(2);

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Tensor& bottom_meta_data = context->input(3);
    auto meta_data = bottom_meta_data.flat<T>();

    const Tensor& bottom_gt = context->input(4);
    const float* gt = bottom_gt.flat<float>().data();

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_label.dims() == 3,
                errors::InvalidArgument("label must be 3-dimensional"));

    OP_REQUIRES(context, bottom_vertex.dims() == 4,
                errors::InvalidArgument("vertex must be 4-dimensional"));

    // batch size
    int batch_size = bottom_label.dim_size(0);
    // height
    int height = bottom_label.dim_size(1);
    // width
    int width = bottom_label.dim_size(2);
    // num of classes
    int num_classes = bottom_vertex.dim_size(3) / VERTEX_CHANNELS;
    int num_meta_data = bottom_meta_data.dim_size(3);
    int num_gt = bottom_gt.dim_size(0);

    // for each image, run hough voting
    std::vector<cv::Vec<float, 14> > outputs;
    const float* extents = bottom_extents.flat<float>().data();

    // bb3Ds
    std::vector<std::vector<cv::Point3f>> bb3Ds;
    getBb3Ds(extents, bb3Ds, num_classes);

    int index_meta_data = 0;
    float fx, fy, px, py;
    for (int n = 0; n < batch_size; n++)
    {
      const int* labelmap = bottom_label.flat<int>().data() + n * height * width;
      const float* vertmap = bottom_vertex.flat<float>().data() + n * height * width * VERTEX_CHANNELS * num_classes;
      fx = meta_data(index_meta_data + 0);
      fy = meta_data(index_meta_data + 4);
      px = meta_data(index_meta_data + 2);
      py = meta_data(index_meta_data + 5);
      hough_voting(labelmap, vertmap, bb3Ds, n, height, width, num_classes, is_train_, fx, fy, px, py, outputs);
      index_meta_data += num_meta_data;
    }

    if (outputs.size() == 0)
    {
      std::cout << "no detection" << std::endl;
      // add a dummy detection to the output
      cv::Vec<float, 14> roi;
      roi(0) = 0;
      roi(1) = -1;
      outputs.push_back(roi);
    }

    // Create output tensors
    // top_box
    int dims[2];
    dims[0] = outputs.size();
    dims[1] = 7;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape);

    Tensor* top_box_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_box_tensor));
    float* top_box = top_box_tensor->template flat<float>().data();

    // top_pose
    dims[1] = 7;
    TensorShape output_shape_pose;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape_pose);

    Tensor* top_pose_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_pose, &top_pose_tensor));
    float* top_pose = top_pose_tensor->template flat<float>().data();

    // top target
    dims[1] = 4 * num_classes;
    TensorShape output_shape_target;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape_target);

    Tensor* top_target_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_target, &top_target_tensor));
    float* top_target = top_target_tensor->template flat<float>().data();
    memset(top_target, 0, outputs.size() * 4 * num_classes *sizeof(T));

    // top weight
    Tensor* top_weight_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, output_shape_target, &top_weight_tensor));
    float* top_weight = top_weight_tensor->template flat<float>().data();
    memset(top_weight, 0, outputs.size() * 4 * num_classes *sizeof(T));

    // top domain
    int num = outputs.size();
    TensorShape output_shape_domain;
    TensorShapeUtils::MakeShape(&num, 1, &output_shape_domain);
    Tensor* top_domain_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, output_shape_domain, &top_domain_tensor));
    int* top_domain = top_domain_tensor->template flat<int>().data();
    memset(top_domain, 0, outputs.size() * sizeof(int));
    
    for(int n = 0; n < outputs.size(); n++)
    {
      cv::Vec<float, 14> roi = outputs[n];

      for (int i = 0; i < 7; i++)
        top_box[n * 7 + i] = roi(i);

      for (int i = 0; i < 7; i++)
        top_pose[n * 7 + i] = roi(7 + i);

      if (num_gt == 0)
        top_domain[n] = 1;
      else
        top_domain[n] = 0;
    }

    if (is_train_)
      compute_target_weight(height, width, top_target, top_weight, bb3Ds, gt, num_gt, num_classes, fx, fy, px, py, outputs);
  }
 private:
  int is_train_;
  float threshold_vote_;
  float threshold_percentage_;
  int skip_pixels_;
};

REGISTER_KERNEL_BUILDER(Name("Houghvotinggpu").Device(DEVICE_CPU).TypeConstraint<float>("T"), HoughvotinggpuOp<CPUDevice, float>);

template <class T>
class HoughvotinggpuOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit HoughvotinggpuOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_train", &is_train_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, is_train_ >= 0,
                errors::InvalidArgument("Need is_train >= 0, got ",
                                        is_train_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold_vote", &threshold_vote_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("threshold_percentage", &threshold_percentage_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_pixels", &skip_pixels_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_label = context->input(0);
    const Tensor& bottom_vertex = context->input(1);

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_label.dims() == 3,
                errors::InvalidArgument("label must be 3-dimensional"));

    OP_REQUIRES(context, bottom_vertex.dims() == 4,
                errors::InvalidArgument("vertex must be 4-dimensional"));

    const Tensor& bottom_extents = context->input(2);
    const float* extents = bottom_extents.flat<float>().data();

    // format of the meta_data
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Tensor& bottom_meta_data = context->input(3);

    const Tensor& bottom_gt = context->input(4);
    const float* gt = bottom_gt.flat<float>().data();

    int batch_size = bottom_label.dim_size(0);
    int height = bottom_label.dim_size(1);
    int width = bottom_label.dim_size(2);
    int num_classes = bottom_vertex.dim_size(3) / VERTEX_CHANNELS;
    int num_meta_data = bottom_meta_data.dim_size(3);
    int num_gt = bottom_gt.dim_size(0);

    float inlierThreshold = 0.9;
    int labelThreshold = 500;
    Tensor top_box_tensor_tmp, top_pose_tensor_tmp, top_target_tensor_tmp, top_weight_tensor_tmp, top_domain_tensor_tmp, num_rois_tensor_tmp;
    allocate_outputs(context, &top_box_tensor_tmp, &top_pose_tensor_tmp, &top_target_tensor_tmp, &top_weight_tensor_tmp, 
      &top_domain_tensor_tmp, &num_rois_tensor_tmp, num_classes);
    float* top_box = top_box_tensor_tmp.flat<float>().data();
    float* top_pose = top_pose_tensor_tmp.flat<float>().data();
    float* top_target = top_target_tensor_tmp.flat<float>().data();
    float* top_weight = top_weight_tensor_tmp.flat<float>().data();
    int* top_domain = top_domain_tensor_tmp.flat<int>().data();
    int* num_rois_device = num_rois_tensor_tmp.flat<int>().data();
    reset_outputs(top_box, top_pose, top_target, top_weight, top_domain, num_rois_device, num_classes);

    for (int n = 0; n < batch_size; n++)
    {
      const int* labelmap = bottom_label.flat<int>().data() + n * height * width;
      const float* vertmap = bottom_vertex.flat<float>().data() + n * height * width * VERTEX_CHANNELS * num_classes;
      const float* meta_data = bottom_meta_data.flat<float>().data() + n * num_meta_data;
      HoughVotingLaucher(context, labelmap, vertmap, extents, meta_data, gt, n, batch_size, height, width, num_classes, num_gt,
        is_train_, inlierThreshold, labelThreshold, threshold_vote_, threshold_percentage_, skip_pixels_,
        top_box, top_pose, top_target, top_weight, top_domain, num_rois_device, context->eigen_device<Eigen::GpuDevice>());
    }

    int num_rois;
    copy_num_rois(&num_rois, num_rois_device);
    // dummy output
    if (num_rois == 0)
      num_rois = 1;

    // Create output tensors
    // top_box
    int dims[2];
    dims[0] = num_rois;
    dims[1] = 7;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape);

    Tensor* top_box_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_box_tensor));
    float* top_box_final = top_box_tensor->flat<float>().data();

    // top_pose
    dims[1] = 7;
    TensorShape output_shape_pose;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape_pose);

    Tensor* top_pose_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_pose, &top_pose_tensor));
    float* top_pose_final = top_pose_tensor->flat<float>().data();

    // top target
    dims[1] = 4 * num_classes;
    TensorShape output_shape_target;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape_target);

    Tensor* top_target_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_target, &top_target_tensor));
    float* top_target_final = top_target_tensor->flat<float>().data();

    // top weight
    Tensor* top_weight_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, output_shape_target, &top_weight_tensor));
    float* top_weight_final = top_weight_tensor->flat<float>().data();

    // top domain
    TensorShape output_shape_domain;
    TensorShapeUtils::MakeShape(&num_rois, 1, &output_shape_domain);
    Tensor* top_domain_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, output_shape_domain, &top_domain_tensor));
    int* top_domain_final = top_domain_tensor->template flat<int>().data();

    copy_outputs(top_box, top_pose, top_target, top_weight, top_domain, top_box_final, 
      top_pose_final, top_target_final, top_weight_final, top_domain_final, num_classes, num_rois);
  }
 private:
  int is_train_;
  float threshold_vote_;
  float threshold_percentage_;
  int skip_pixels_;
};

REGISTER_KERNEL_BUILDER(Name("Houghvotinggpu").Device(DEVICE_GPU).TypeConstraint<float>("T"), HoughvotinggpuOp<Eigen::GpuDevice, float>);

// compute gradient
template <class Device, class T>
class HoughvotinggpuGradOp : public OpKernel {
 public:
  explicit HoughvotinggpuGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_label = context->input(0);
    const Tensor& bottom_vertex = context->input(1);

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_label.dims() == 3,
                errors::InvalidArgument("label must be 3-dimensional"));

    OP_REQUIRES(context, bottom_vertex.dims() == 4,
                errors::InvalidArgument("vertex must be 4-dimensional"));

    // batch size
    int batch_size = bottom_label.dim_size(0);
    // height
    int height = bottom_label.dim_size(1);
    // width
    int width = bottom_label.dim_size(2);
    // num of classes
    int num_classes = bottom_vertex.dim_size(3) / VERTEX_CHANNELS;

    // construct the output shape
    TensorShape output_shape = bottom_label.shape();
    Tensor* top_label_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_label_tensor));
    float* top_label = top_label_tensor->flat<float>().data();

    TensorShape output_shape_1 = bottom_vertex.shape();
    Tensor* top_vertex_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_1, &top_vertex_tensor));
    float* top_vertex = top_vertex_tensor->flat<float>().data();

    set_gradients(top_label, top_vertex, batch_size, height, width, num_classes);
  }
};

// REGISTER_KERNEL_BUILDER(Name("HoughvotinggpuGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), HoughvotinggpuGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("HoughvotinggpuGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), HoughvotinggpuGradOp<Eigen::GpuDevice, float>);

void hough_voting(const int* labelmap, const float* vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds, 
  int batch, int height, int width, int num_classes, int is_train,
  float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> >& outputs)
{
  float inlierThreshold = 0.9;
  int votingThreshold = 50;

  // camera matrix
  cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
  camMat(0, 0) = fx;
  camMat(1, 1) = fy;
  camMat(2, 2) = 1.f;
  camMat(0, 2) = px;
  camMat(1, 2) = py;

  // initialize hough space
  int* hough_space = (int*)malloc(sizeof(int) * height * width * num_classes);
  memset(hough_space, 0, height * width * num_classes);

  int* flags = (int*)malloc(sizeof(int) * num_classes);
  memset(flags, 0, num_classes);

  // for each pixel
  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      int c = labelmap[y * width + x];
      if (c > 0)
      {
        flags[c] = 1;
        // read the predict center direction
        int offset = VERTEX_CHANNELS * c + VERTEX_CHANNELS * num_classes * (y * width + x);
        float u = vertmap[offset];
        float v = vertmap[offset + 1];
        float norm = sqrt(u * u + v * v);
        u /= norm;
        v /= norm;

        // voting
	float delta = 1.0 / fabs(u);
        float cx = x;
        float cy = y;
        while(1)
        {
          cx += delta * u;
          cy += delta * v;
          int center_x = int(cx);
          int center_y = int(cy);
          if (center_x >= 0 && center_x < width && center_y >= 0 && center_y < height)
          {
            offset = c + num_classes * (center_y * width + center_x);
            hough_space[offset] += 1;
          }
          else
            break;
        }
      }
    }
  }

  // find the maximum in hough space
  for (int c = 1; c < num_classes; c++)
  {
    if (flags[c])
    {
      int max_vote = 0;
      int max_x, max_y;
      for (int x = 0; x < width; x++)
      {
        for (int y = 0; y < height; y++)
        {
          int offset = c + num_classes * (y * width + x);
          if (hough_space[offset] > max_vote)
          {
            max_vote = hough_space[offset];
            max_x = x;
            max_y = y;
          }
        }
      }
      if (max_vote < votingThreshold)
        continue;

      // center
      cv::Point2f center(max_x, max_y);
      int bb_width, bb_height;
      float bb_distance;
      compute_width_height(labelmap, vertmap, center, bb3Ds, camMat, inlierThreshold, height, width, c, num_classes, bb_width, bb_height, bb_distance);

      // construct output
      cv::Vec<float, 14> roi;
      roi(0) = batch;
      roi(1) = c;

      // bounding box
      float scale = 0.05;
      roi(2) = center.x - bb_width * (0.5 + scale);
      roi(3) = center.y - bb_height * (0.5 + scale);
      roi(4) = center.x + bb_width * (0.5 + scale);
      roi(5) = center.y + bb_height * (0.5 + scale);

      // score
      roi(6) = max_vote;

      // pose
      float rx = (center.x - px) / fx;
      float ry = (center.y - py) / fy;
      roi(7) = 1;
      roi(8) = 0;
      roi(9) = 0;
      roi(10) = 0;
      roi(11) = rx * bb_distance;
      roi(12) = ry * bb_distance;
      roi(13) = bb_distance;

      outputs.push_back(roi);

      if (is_train)
      {
        // add jittering rois
        float x1 = roi(2);
        float y1 = roi(3);
        float x2 = roi(4);
        float y2 = roi(5);
        float ww = x2 - x1;
        float hh = y2 - y1;

        // (-1, -1)
        roi(2) = x1 - 0.05 * ww;
        roi(3) = y1 - 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (+1, -1)
        roi(2) = x1 + 0.05 * ww;
        roi(3) = y1 - 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (-1, +1)
        roi(2) = x1 - 0.05 * ww;
        roi(3) = y1 + 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (+1, +1)
        roi(2) = x1 + 0.05 * ww;
        roi(3) = y1 + 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (0, -1)
        roi(2) = x1;
        roi(3) = y1 - 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (-1, 0)
        roi(2) = x1 - 0.05 * ww;
        roi(3) = y1;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (0, +1)
        roi(2) = x1;
        roi(3) = y1 + 0.05 * hh;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);

        // (+1, 0)
        roi(2) = x1 + 0.05 * ww;
        roi(3) = y1;
        roi(4) = roi(2) + ww;
        roi(5) = roi(3) + hh;
        outputs.push_back(roi);
      }
    }
  }
}

inline float angle_distance(cv::Point2f x, cv::Point2f n, cv::Point2f p)
{
  return n.dot(x - p) / (cv::norm(n) * cv::norm(x - p));
}

inline void compute_width_height(const int* labelmap, const float* vertmap, cv::Point2f center, 
  std::vector<std::vector<cv::Point3f>> bb3Ds, cv::Mat camMat, float inlierThreshold, 
  int height, int width, int channel, int num_classes, int & bb_width, int & bb_height, float & bb_distance)
{
  float d = 0;
  int count = 0;

  // for each pixel
  std::vector<float> dx;
  std::vector<float> dy;
  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      if (labelmap[y * width + x] == channel)
      {
        cv::Point2f point(x, y);
  
        // read out object coordinate
        int offset = VERTEX_CHANNELS * channel + VERTEX_CHANNELS * num_classes * (y * width + x);
        float u = vertmap[offset];
        float v = vertmap[offset + 1];
        float distance = exp(vertmap[offset + 2]);
        float norm = sqrt(u * u + v * v);
        u /= norm;
        v /= norm;
        cv::Point2f direction(u, v);

        // inlier check
        if(angle_distance(center, direction, point) > inlierThreshold)
        {
          dx.push_back(fabs(point.x - center.x));
          dy.push_back(fabs(point.y - center.y));
          d += distance;
          count++;
        }
      }
    }
  }
  bb_distance = d / count;

  // estimate a projection
  cv::Mat tvec(3, 1, CV_64F);
  cv::Mat rvec(3, 1, CV_64F);
  for(int i = 0; i < 3; i++)
  {
    tvec.at<double>(i, 0) = 0;
    rvec.at<double>(i, 0) = 0;
  }
  tvec.at<double>(2, 0) = bb_distance;
  std::vector<cv::Point2f> bb2D;
  cv::projectPoints(bb3Ds[channel-1], rvec, tvec, camMat, cv::Mat(), bb2D);
    
  // get min-max of projected vertices
  int minX = 1e8;
  int maxX = -1e8;
  int minY = 1e8;
  int maxY = -1e8;
  for(int i = 0; i < bb2D.size(); i++)
  {
    minX = std::min((float) minX, bb2D[i].x);
    minY = std::min((float) minY, bb2D[i].y);
    maxX = std::max((float) maxX, bb2D[i].x);
    maxY = std::max((float) maxY, bb2D[i].y);
  }
  cv::Rect bb = cv::Rect(0, 0, (maxX - minX + 1), (maxY - minY + 1));

  std::vector<float>::iterator it;
  it = std::remove_if(dx.begin(), dx.end(), std::bind2nd(std::greater<float>(), std::max(bb.width, bb.height) ));
  dx.erase(it, dx.end()); 

  it = std::remove_if(dy.begin(), dy.end(), std::bind2nd(std::greater<float>(), std::max(bb.width, bb.height) ));
  dy.erase(it, dy.end()); 

  std::sort(dx.begin(), dx.end());
  std::sort(dy.begin(), dy.end());

  bb_width = 2 * dx[int(dx.size() * 0.95)];
  bb_height = 2 * dy[int(dy.size() * 0.95)];
}


// compute the pose target and weight
void compute_target_weight(int height, int width, float* target, float* weight, std::vector<std::vector<cv::Point3f>> bb3Ds, 
  const float* poses_gt, int num_gt, int num_classes, float fx, float fy, float px, float py, std::vector<cv::Vec<float, 14> > outputs)
{
  int num = outputs.size();
  float threshold = 0.2;

  // camera matrix
  cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
  camMat(0, 0) = fx;
  camMat(1, 1) = fy;
  camMat(2, 2) = 1.f;
  camMat(0, 2) = px;
  camMat(1, 2) = py;

  // compute the gt boxes
  std::vector<cv::Rect> bb2Ds_gt(num_gt);
  for (int i = 0; i < num_gt; i++)
  {
    Eigen::Quaternionf quaternion(poses_gt[i * 13 + 6], poses_gt[i * 13 + 7], poses_gt[i * 13 + 8], poses_gt[i * 13 + 9]);
    Eigen::Matrix3f rmatrix = quaternion.toRotationMatrix();
    cv::Mat rmat_trans = cv::Mat(3, 3, CV_32F, rmatrix.data());
    cv::Mat rmat;
    cv::transpose(rmat_trans, rmat);
    cv::Mat rvec(3, 1, CV_64F);
    cv::Rodrigues(rmat, rvec);
    cv::Mat tvec(3, 1, CV_64F);
    tvec.at<double>(0, 0) = poses_gt[i * 13 + 10];
    tvec.at<double>(1, 0) = poses_gt[i * 13 + 11];
    tvec.at<double>(2, 0) = poses_gt[i * 13 + 12];

    int objID = int(poses_gt[i * 13 + 1]);
    std::vector<cv::Point3f> bb3D = bb3Ds[objID-1];
    bb2Ds_gt[i] = getBB2D(width, height, bb3D, camMat, rvec, tvec);
  }

  for (int i = 0; i < num; i++)
  {
    cv::Vec<float, 14> roi = outputs[i];
    int batch_id = int(roi(0));
    int class_id = int(roi(1));

    // find the gt index
    int gt_ind = -1;
    for (int j = 0; j < num_gt; j++)
    {
      int gt_batch = int(poses_gt[j * 13 + 0]);
      int gt_id = int(poses_gt[j * 13 + 1]);
      if(class_id == gt_id && batch_id == gt_batch)
      {
        gt_ind = j;
        break;
      }
    }

    if (gt_ind == -1)
      continue;

    // compute bounding box overlap
    float x1 = roi(2);
    float y1 = roi(3);
    float x2 = roi(4);
    float y2 = roi(5);
    cv::Rect bb2D(x1, y1, x2-x1, y2-y1);

    float overlap = getIoU(bb2D, bb2Ds_gt[gt_ind]);
    if (overlap < threshold)
      continue;

    target[i * 4 * num_classes + 4 * class_id + 0] = poses_gt[gt_ind * 13 + 6];
    target[i * 4 * num_classes + 4 * class_id + 1] = poses_gt[gt_ind * 13 + 7];
    target[i * 4 * num_classes + 4 * class_id + 2] = poses_gt[gt_ind * 13 + 8];
    target[i * 4 * num_classes + 4 * class_id + 3] = poses_gt[gt_ind * 13 + 9];

    weight[i * 4 * num_classes + 4 * class_id + 0] = 1;
    weight[i * 4 * num_classes + 4 * class_id + 1] = 1;
    weight[i * 4 * num_classes + 4 * class_id + 2] = 1;
    weight[i * 4 * num_classes + 4 * class_id + 3] = 1;
  }
}


// get 3D bounding boxes
void getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
{
  // for each object
  for (int i = 1; i < num_classes; i++)
  {
    cv::Vec<float, 3> extent;
    extent(0) = extents[i * 3];
    extent(1) = extents[i * 3 + 1];
    extent(2) = extents[i * 3 + 2];

    bb3Ds.push_back(getBB3D(extent));
  }
}


inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent)
{
  std::vector<cv::Point3f> bb;  
  float xHalf = extent[0] * 0.5;
  float yHalf = extent[1] * 0.5;
  float zHalf = extent[2] * 0.5;
    
  bb.push_back(cv::Point3f(xHalf, yHalf, zHalf));
  bb.push_back(cv::Point3f(-xHalf, yHalf, zHalf));
  bb.push_back(cv::Point3f(xHalf, -yHalf, zHalf));
  bb.push_back(cv::Point3f(-xHalf, -yHalf, zHalf));
    
  bb.push_back(cv::Point3f(xHalf, yHalf, -zHalf));
  bb.push_back(cv::Point3f(-xHalf, yHalf, -zHalf));
  bb.push_back(cv::Point3f(xHalf, -yHalf, -zHalf));
  bb.push_back(cv::Point3f(-xHalf, -yHalf, -zHalf));
    
  return bb;
}


inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& rvec, const cv::Mat& tvec)
{    
  // project 3D bounding box vertices into the image
  std::vector<cv::Point2f> bb2D;
  cv::projectPoints(bb3D, rvec, tvec, camMat, cv::Mat(), bb2D);
    
  // get min-max of projected vertices
  int minX = imageWidth - 1;
  int maxX = 0;
  int minY = imageHeight - 1;
  int maxY = 0;
    
  for(unsigned j = 0; j < bb2D.size(); j++)
  {
    minX = std::min((float) minX, bb2D[j].x);
    minY = std::min((float) minY, bb2D[j].y);
    maxX = std::max((float) maxX, bb2D[j].x);
    maxY = std::max((float) maxY, bb2D[j].y);
  }
    
  // clamp at image border
  minX = clamp(minX, 0, imageWidth - 1);
  maxX = clamp(maxX, 0, imageWidth - 1);
  minY = clamp(minY, 0, imageHeight - 1);
  maxY = clamp(maxY, 0, imageHeight - 1);
    
  return cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));
}


inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2)
{
  cv::Rect intersection = bb1 & bb2;
  return (intersection.area() / (float) (bb1.area() + bb2.area() - intersection.area()));
}
