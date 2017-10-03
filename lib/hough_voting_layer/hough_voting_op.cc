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

#include "types.h"
#include "sampler2D.h"
#include "ransac.h"
#include "Hypothesis.h"
#include "detection.h"
#include <Eigen/Geometry> 

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Houghvoting")
    .Attr("T: {float, double}")
    .Attr("is_train: int")
    .Input("bottom_label: int32")
    .Input("bottom_vertex: T")
    .Input("bottom_extents: T")
    .Input("bottom_meta_data: T")
    .Input("bottom_gt: T")
    .Output("top_box: T")
    .Output("top_pose: T")
    .Output("top_target: T")
    .Output("top_weight: T");

REGISTER_OP("HoughvotingGrad")
    .Attr("T: {float, double}")
    .Input("bottom_label: int32")
    .Input("bottom_vertex: T")
    .Input("grad: T")
    .Output("output_label: T")
    .Output("output_vertex: T");

/**
 * @brief Data used in NLOpt callback loop.
 */
struct DataForOpt
{
  int imageWidth;
  int imageHeight;
  float rx, ry;
  cv::Rect bb2D;
  std::vector<cv::Point3f> bb3D;
  cv::Mat_<float> camMat;
};

void getLabels(const int* label_map, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea);
void getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
inline bool samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, std::vector<float>& distances, const cv::Point2f& pt2D, const float* vertmap, int width, int num_classes);
std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt);
inline float point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p);
inline void countInliers2D(TransHyp& hyp, const float * vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes, int pixelBatch);
inline void updateHyp2D(TransHyp& hyp, int maxPixels);
inline void filterInliers2D(TransHyp& hyp, int maxInliers);
inline cv::Point2f getMode2D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, float & distance, int width, int num_classes);
static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data);
double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations);
void estimateCenter(const int* labelmap, const float* vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds, int batch, int height, int width, int num_classes, int is_train,
  float fx, float fy, float px, float py, std::vector<cv::Vec<float, 13> >& outputs);
void compute_target_weight(int height, int width, float* target, float* weight, std::vector<std::vector<cv::Point3f>> bb3Ds, const float* poses_gt, int num_gt, int num_classes, float fx, float fy, float px, float py, std::vector<cv::Vec<float, 13> > outputs);
inline void compute_width_height(TransHyp& hyp, const float* vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes);

template <typename Device, typename T>
class HoughvotingOp : public OpKernel {
 public:
  explicit HoughvotingOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_train", &is_train_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, is_train_ >= 0,
                errors::InvalidArgument("Need is_train >= 0, got ",
                                        is_train_));
  }

  // bottom_label: (batch_size, height, width)
  // bottom_vertex: (batch_size, height, width, 3 * num_classes)
  // top_box: (num, 6) i.e., batch_index, cls, x1, y1, x2, y2
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
    std::vector<cv::Vec<float, 13> > outputs;
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

      estimateCenter(labelmap, vertmap, bb3Ds, n, height, width, num_classes, is_train_, fx, fy, px, py, outputs);

      index_meta_data += num_meta_data;
    }

    if (outputs.size() == 0)
    {
      std::cout << "no detection" << std::endl;
      // add a dummy detection to the output
      cv::Vec<float, 13> roi;
      roi(0) = 0;
      roi(1) = -1;
      roi(2) = 0;
      roi(3) = 0;
      roi(4) = 1;
      roi(5) = 1;
      roi(6) = 1;
      outputs.push_back(roi);
    }

    // Create output tensors
    // top_box
    int dims[2];
    dims[0] = outputs.size();
    dims[1] = 6;
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
    
    for(int n = 0; n < outputs.size(); n++)
    {
      cv::Vec<float, 13> roi = outputs[n];

      for (int i = 0; i < 6; i++)
        top_box[n * 6 + i] = roi(i);

      for (int i = 0; i < 7; i++)
        top_pose[n * 7 + i] = roi(6 + i);
    }

    if (is_train_)
      compute_target_weight(height, width, top_target, top_weight, bb3Ds, gt, num_gt, num_classes, fx, fy, px, py, outputs);
  }
 private:
  int is_train_;
};

REGISTER_KERNEL_BUILDER(Name("Houghvoting").Device(DEVICE_CPU).TypeConstraint<float>("T"), HoughvotingOp<CPUDevice, float>);


// compute gradient
template <class Device, class T>
class HoughvotingGradOp : public OpKernel {
 public:
  explicit HoughvotingGradOp(OpKernelConstruction* context) : OpKernel(context) {
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
    T* top_label = top_label_tensor->template flat<T>().data();
    memset(top_label, 0, batch_size * height * width * sizeof(T));

    TensorShape output_shape_1 = bottom_vertex.shape();
    Tensor* top_vertex_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_1, &top_vertex_tensor));
    T* top_vertex = top_vertex_tensor->template flat<T>().data();
    memset(top_vertex, 0, batch_size * height * width * 2 * num_classes *sizeof(T));
  }
};

REGISTER_KERNEL_BUILDER(Name("HoughvotingGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), HoughvotingGradOp<CPUDevice, float>);


// get label lists
void getLabels(const int* label_map, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea)
{
  for(int i = 0; i < num_classes; i++)
    labels.push_back( std::vector<int>() );

  // for each pixel
  #pragma omp parallel for
  for(int x = 0; x < width; x++)
  for(int y = 0; y < height; y++)
  {
    int label = label_map[y * width + x];
    labels[label].push_back(y * width + x);
  }

  for(int i = 1; i < num_classes; i++)
  {
    if (labels[i].size() > minArea)
    {
      object_ids.push_back(i);
    }
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


inline cv::Point2f getMode2D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, float & distance, int width, int num_classes)
{
  int channel = VERTEX_CHANNELS * objID;
  int offset = channel + VERTEX_CHANNELS * num_classes * (pt.y * width + pt.x);

  jp::coord2_t mode;
  mode(0) = vertmap[offset];
  mode(1) = vertmap[offset + 1];
  distance = vertmap[offset + 2];

  return cv::Point2f(mode(0), mode(1));
}


inline bool samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, std::vector<float>& distances, const cv::Point2f& pt2D, const float* vertmap, int width, int num_classes)
{
  float distance;
  cv::Point2f obj = getMode2D(objID, pt2D, vertmap, distance, width, num_classes); // read out object coordinate

  eyePts.push_back(pt2D);
  objPts.push_back(obj);
  distances.push_back(distance);

  if (distance < 0)
    return false;
  else
    return true;
}


/**
 * @brief Creates a list of pose hypothesis (potentially belonging to multiple objects) which still have to be processed (e.g. refined).
 * 
 * The method includes all remaining hypotheses of an object if there is still more than one, or if there is only one remaining but it still needs to be refined.
 * 
 * @param hypMap Map of object ID to a list of hypotheses for that object.
 * @param maxIt Each hypotheses should be at least this often refined.
 * @return std::vector< Ransac3D::TransHyp*, std::allocator< void > > List of hypotheses to be processed further.
*/
std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt, int is_train)
{
  std::vector<TransHyp*> workingQueue;

  if (is_train)
  {      
    for(auto it = hypMap.begin(); it != hypMap.end(); it++)
    for(int h = 0; h < it->second.size(); h++)
      if(it->second[h].refSteps < maxIt)
        workingQueue.push_back(&(it->second[h]));
  }
  else
  {
    for(auto it = hypMap.begin(); it != hypMap.end(); it++)
    for(int h = 0; h < it->second.size(); h++)
      if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
        workingQueue.push_back(&(it->second[h]));
  }

  return workingQueue;
}


inline float point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p)
{
  float n1 = -n.y;
  float n2 = n.x;
  float p1 = p.x;
  float p2 = p.y;
  float x1 = x.x;
  float x2 = x.y;

  return fabs(n1 * (x1 - p1) + n2 * (x2 - p2)) / sqrt(n1 * n1 + n2 * n2);
}


inline float angle_distance(cv::Point2f x, cv::Point2f n, cv::Point2f p)
{
  return n.dot(x - p);
}


inline void countInliers2D(TransHyp& hyp, const float * vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes, int pixelBatch)
{
  // reset data of last RANSAC iteration
  hyp.inlierPts2D.clear();
  hyp.inliers = 0;

  hyp.effPixels = 0; // num of pixels drawn
  hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	

  int maxPt = labels[hyp.objID].size(); // num of pixels of this class
  float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel

  std::mt19937 generator;
  std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept

  for(unsigned ptIdx = 0; ptIdx < maxPt;)
  {
    int index = labels[hyp.objID][ptIdx];
    cv::Point2d pt2D(index % width, index / width);
  
    hyp.effPixels++;
  
    // read out object coordinate
    float distance;
    cv::Point2d obj = getMode2D(hyp.objID, pt2D, vertmap, distance, width, num_classes);

    // inlier check
    float d = cv::norm(hyp.center - pt2D);
    if(point2line(hyp.center, obj, pt2D) < inlierThreshold && angle_distance(hyp.center, obj, pt2D) > 0 && d < std::max(hyp.bb.width, hyp.bb.height))
    {
      hyp.inlierPts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(obj, pt2D)); // store object coordinate - camera coordinate correspondence
      hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
    }

    // advance to the next accepted pixel
    if(successRate < 1)
      ptIdx += std::max(1, distribution(generator));
    else
      ptIdx++;
  }
}


inline void compute_width_height(TransHyp& hyp, const float* vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes)
{
  float w = -1;
  float h = -1;
  int maxPt = labels[hyp.objID].size(); // num of pixels of this class

  for(unsigned ptIdx = 0; ptIdx < maxPt; ptIdx++)
  {
    int index = labels[hyp.objID][ptIdx];
    cv::Point2d pt2D(index % width, index / width);
  
    // read out object coordinate
    float distance;
    cv::Point2d obj = getMode2D(hyp.objID, pt2D, vertmap, distance, width, num_classes);

    // inlier check
    float d = cv::norm(hyp.center - pt2D);
    if(point2line(hyp.center, obj, pt2D) < inlierThreshold && angle_distance(hyp.center, obj, pt2D) > 0 && d < std::max(hyp.bb.width, hyp.bb.height))
    {
      float x = fabs(pt2D.x - hyp.center.x);
      float y = fabs(pt2D.y - hyp.center.y);
      if (x > w)
        w = x;
      if (y > h)
        h = y;
    }
  }
  hyp.width_ = 2 * w;
  hyp.height_ = 2 * h;
}


inline void updateHyp2D(TransHyp& hyp, int maxPixels)
{
  if(hyp.inlierPts2D.size() < 4) return;
  filterInliers2D(hyp, maxPixels); // limit the number of correspondences
      
  // data conversion
  cv::Point2d center = hyp.center;
  Hypothesis trans(center);	
	
  // recalculate pose
  trans.calcCenter(hyp.inlierPts2D);
  hyp.center = trans.getCenter();
}


inline void filterInliers2D(TransHyp& hyp, int maxInliers)
{
  if(hyp.inlierPts2D.size() < maxInliers) return; // maximum number not reached, do nothing
      		
  std::vector<std::pair<cv::Point2d, cv::Point2d>> inlierPts; // filtered list of inlier correspondences
	
  // select random correspondences to keep
  for(unsigned i = 0; i < maxInliers; i++)
  {
    int idx = irand(0, hyp.inlierPts2D.size());
	    
    inlierPts.push_back(hyp.inlierPts2D[idx]);
  }
	
  hyp.inlierPts2D = inlierPts;
}


void estimateCenter(const int* labelmap, const float* vertmap, std::vector<std::vector<cv::Point3f>> bb3Ds, int batch, int height, int width, int num_classes, int is_train,
  float fx, float fy, float px, float py, std::vector<cv::Vec<float, 13> >& outputs)
{     
  //set parameters, see documentation of GlobalProperties
  int maxIterations = 10000000;
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)
  float minDist2D = 10;
  float inlierThreshold3D = 0.5;
  int ransacIterations;  // 256
  int poseIterations = 100;
  int preemptiveBatch;  // 1000
  int maxPixels;  // 1000
  int refIt;  // 8

  if (is_train)
  {
    ransacIterations = 256;
    preemptiveBatch = 100;
    maxPixels = 1000;
    refIt = 4;
  }
  else
  {
    ransacIterations = 256 * 1;
    preemptiveBatch = 100 * 1;
    maxPixels = 1000 * 1;
    refIt = 8;
  }

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(labelmap, labels, object_ids, width, height, num_classes, minArea);

  // camera matrix
  cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
  camMat(0, 0) = fx;
  camMat(1, 1) = fy;
  camMat(2, 2) = 1.f;
  camMat(0, 2) = px;
  camMat(1, 2) = py;

  if (object_ids.size() == 0)
    return;
	
  int imageWidth = width;
  int imageHeight = height;
		
  // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
  std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
  // sample initial pose hypotheses
  #pragma omp parallel for
  for(unsigned h = 0; h < ransacIterations; h++)
  for(unsigned i = 0; i < maxIterations; i++)
  {
    // camera coordinate - object coordinate correspondences
    std::vector<cv::Point2f> eyePts;
    std::vector<cv::Point2f> objPts;
    std::vector<float> distances;
	    
    // sample first point and choose object ID
    jp::id_t objID = object_ids[irand(0, object_ids.size())];

    if(objID == 0) continue;

    int pindex = irand(0, labels[objID].size());
    int index = labels[objID][pindex];
    cv::Point2f pt1(index % width, index / width);
    
    // sample first correspondence
    if(!samplePoint2D(objID, eyePts, objPts, distances, pt1, vertmap, width, num_classes))
      continue;

    // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt2(index % width, index / width);

    if (cv::norm(pt1 - pt2) < minDist2D)
      continue;

    if(!samplePoint2D(objID, eyePts, objPts, distances, pt2, vertmap, width, num_classes))
      continue;

    // reconstruct
    std::vector<std::pair<cv::Point2d, cv::Point2d>> pts2D;
    float distance = 0;
    for(unsigned j = 0; j < eyePts.size(); j++)
    {
      pts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(
      cv::Point2d(objPts[j].x, objPts[j].y),
      cv::Point2d(eyePts[j].x, eyePts[j].y)
      ));
      distance += distances[j];
    }
    distance /= distances.size();

    Hypothesis trans(pts2D);

    // center
    cv::Point2d center = trans.getCenter();
    int x = int(center.x);
    int y = int(center.y);
    if (num_classes > 2 && x >= 0 && x < width && y >= 0 && y < height)
    {
      if (labelmap[y * width + x] == 0)
        continue;
    }
    
    // create a hypothesis object to store meta data
    TransHyp hyp(objID, center);

    // estimate a projection
    cv::Mat tvec(3, 1, CV_64F);
    cv::Mat rvec(3, 1, CV_64F);
    for(int j = 0; j < 3; j++)
    {
      tvec.at<double>(j, 0) = 0;
      rvec.at<double>(j, 0) = 0;
    }
    tvec.at<double>(2, 0) = distance;
    jp::cv_trans_t pose(rvec, tvec);

    std::vector<cv::Point2f> bb2D;
    cv::projectPoints(bb3Ds[objID-1], pose.first, pose.second, camMat, cv::Mat(), bb2D);
    
    // get min-max of projected vertices
    int minX = 10000000;
    int maxX = -10000000;
    int minY = 10000000;
    int maxY = -10000000;
    
    for(unsigned j = 0; j < bb2D.size(); j++)
    {
	minX = std::min((float) minX, bb2D[j].x);
	minY = std::min((float) minY, bb2D[j].y);
	maxX = std::max((float) maxX, bb2D[j].x);
	maxY = std::max((float) maxY, bb2D[j].y);
    }
    hyp.bb = cv::Rect(0, 0, (maxX - minX + 1), (maxY - minY + 1));

    cv::Point2f c;
    c.x = center.x;
    c.y = center.y;
    if (cv::norm(pt1 - c) > std::max(hyp.bb.width, hyp.bb.height) || cv::norm(pt2 - c) > std::max(hyp.bb.width, hyp.bb.height))
      continue;
    
    #pragma omp critical
    {
      hypMap[objID].push_back(hyp);
    }

    break;
  }

  // create a list of all objects where hyptheses have been found
  std::vector<jp::id_t> objList;
  for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
  {
    objList.push_back(hypPair.first);
  }

  // create a working queue of all hypotheses to process
  std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt, is_train);
	
  // main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
  while(!workingQueue.empty())
  {
    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
      countInliers2D(*(workingQueue[h]), vertmap, labels, inlierThreshold3D, width, num_classes, preemptiveBatch);
	    	    
    // sort hypothesis according to inlier count and discard bad half
    #pragma omp parallel for 
    for(unsigned o = 0; o < objList.size(); o++)
    {
      jp::id_t objID = objList[o];
      if(hypMap[objID].size() > 1)
      {
	std::sort(hypMap[objID].begin(), hypMap[objID].end());
	hypMap[objID].erase(hypMap[objID].begin() + hypMap[objID].size() / 2, hypMap[objID].end());
      }
    }
    workingQueue = getWorkingQueue(hypMap, refIt, is_train);
	    
    // refine
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
    {
      updateHyp2D(*(workingQueue[h]), maxPixels);
      workingQueue[h]->refSteps++;
    }
    
    workingQueue = getWorkingQueue(hypMap, refIt, is_train);
  }

  #pragma omp parallel for
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
  {
    cv::Vec<float, 13> roi;
    roi(0) = batch;
    roi(1) = it->second[h].objID;

    // backproject the center
    cv::Point2d center = it->second[h].center;
    float rx = (center.x - px) / fx;
    float ry = (center.y - py) / fy;
    float distance = it->second[h].compute_distance(vertmap, num_classes, width);

    // initial pose
    std::vector<double> vec(6);
    vec[0] = 0.0;
    vec[1] = 0.0;
    vec[2] = 0.0;
    vec[3] = rx * distance;
    vec[4] = ry * distance;
    vec[5] = distance;

    // convert pose to our format
    cv::Mat tvec(3, 1, CV_64F);
    cv::Mat rvec(3, 1, CV_64F);
      
    for(int i = 0; i < 6; i++)
    {
      if(i > 2) 
        tvec.at<double>(i-3, 0) = vec[i];
      else 
        rvec.at<double>(i, 0) = vec[i];
    }
	
    jp::cv_trans_t trans(rvec, tvec);
    jp::jp_trans_t pose = jp::cv2our(trans);

    // convert to quarternion
    cv::Mat pose_t;
    cv::transpose(pose.first, pose_t);
    Eigen::Map<Eigen::Matrix3d> eigenT( (double*)pose_t.data );
    Eigen::Quaterniond quaternion(eigenT);

    compute_width_height(it->second[h], vertmap, labels, inlierThreshold3D, width, num_classes);
    float scale = 0.05;
    roi(2) = center.x - it->second[h].width_ * (0.5 + scale);
    roi(3) = center.y - it->second[h].height_ * (0.5 + scale);
    roi(4) = center.x + it->second[h].width_ * (0.5 + scale);
    roi(5) = center.y + it->second[h].height_ * (0.5 + scale);

    roi(6) = quaternion.w();
    roi(7) = quaternion.x();
    roi(8) = quaternion.y();
    roi(9) = quaternion.z();
    roi(10) = pose.second.x;
    roi(11) = pose.second.y;
    roi(12) = pose.second.z;

    /*
    std::cout << pose.first << std::endl;
    std::cout << eigenT << std::endl;
    std::cout << quaternion.w() << " " << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << std::endl;
    std::cout << pose.second << std::endl;
    
    std::cout << "Inliers: " << it->second[h].inliers;
    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
    std::cout << "Center " << center << std::endl;
    std::cout << "Width: " << it->second[h].width_ << " Height: " << it->second[h].height_ << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << roi << std::endl;
    */

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


static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
{
  DataForOpt* dataForOpt = (DataForOpt*) data;

  cv::Mat tvec(3, 1, CV_64F);
  cv::Mat rvec(3, 1, CV_64F);
      
  for(int i = 0; i < 6; i++)
  {
    if(i > 2) 
      tvec.at<double>(i-3, 0) = pose[i];
    else 
      rvec.at<double>(i, 0) = pose[i];
  }
	
  jp::cv_trans_t trans(rvec, tvec);

  // project the 3D bounding box according to the current pose
  cv::Rect bb2D = getBB2D(dataForOpt->imageWidth, dataForOpt->imageHeight, dataForOpt->bb3D, dataForOpt->camMat, trans);

  // compute IoU between boxes
  float energy = -1 * getIoU(bb2D, dataForOpt->bb2D);

  return energy;
}


double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations) 
{
  // set up optimization algorithm (gradient free)
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 6); 

  // set optimization bounds 
  double rotRange = 180;
  rotRange *= PI / 180;
  double tRangeXY = 0.01;
  double tRangeZ = 0.01; // pose uncertainty is larger in Z direction
	
  std::vector<double> lb(6);
  lb[0] = vec[0]-rotRange; lb[1] = vec[1]-rotRange; lb[2] = vec[2]-rotRange;
  lb[3] = vec[3]-tRangeXY; lb[4] = vec[4]-tRangeXY; lb[5] = vec[5]-tRangeZ;
  opt.set_lower_bounds(lb);
      
  std::vector<double> ub(6);
  ub[0] = vec[0]+rotRange; ub[1] = vec[1]+rotRange; ub[2] = vec[2]+rotRange;
  ub[3] = vec[3]+tRangeXY; ub[4] = vec[4]+tRangeXY; ub[5] = vec[5]+tRangeZ;
  opt.set_upper_bounds(ub);
      
  // configure NLopt
  opt.set_min_objective(optEnergy, &data);
  opt.set_maxeval(iterations);

  // run optimization
  double energy;
  nlopt::result result = opt.optimize(vec, energy);

  // std::cout << "IoU after optimization: " << -energy << std::endl;
   
  return energy;
}


// compute the pose target and weight
void compute_target_weight(int height, int width, float* target, float* weight, std::vector<std::vector<cv::Point3f>> bb3Ds, 
  const float* poses_gt, int num_gt, int num_classes, float fx, float fy, float px, float py, std::vector<cv::Vec<float, 13> > outputs)
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
    cv::Point3d tvec(poses_gt[i * 13 + 10], poses_gt[i * 13 + 11], poses_gt[i * 13 + 12]);
    jp::jp_trans_t pose(rmat, tvec);
    jp::cv_trans_t trans = jp::our2cv(pose);

    int objID = int(poses_gt[i * 13 + 1]);
    std::vector<cv::Point3f> bb3D = bb3Ds[objID-1];
    bb2Ds_gt[i] = getBB2D(width, height, bb3D, camMat, trans);
  }

  for (int i = 0; i < num; i++)
  {
    cv::Vec<float, 13> roi = outputs[i];
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
