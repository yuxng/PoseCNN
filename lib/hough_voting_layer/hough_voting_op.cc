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

#include "types.h"
#include "sampler2D.h"
#include "ransac.h"
#include "Hypothesis.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Houghvoting")
    .Attr("T: {float, double}")
    .Input("bottom_prob: T")
    .Input("bottom_vertex: T")
    .Output("top_box: T");

REGISTER_OP("HoughvotingGrad")
    .Attr("T: {float, double}")
    .Input("bottom_prob: T")
    .Input("bottom_vertex: T")
    .Input("grad: T")
    .Output("output_prob: T")
    .Output("output_vertex: T");

void getProbs(const float* probability, std::vector<jp::img_stat_t>& probs, int width, int height, int num_classes);
void getCenters(const float* vertmap, std::vector<jp::img_center_t>& vertexs, int width, int height, int num_classes);
void getLabels(const float* probability, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea);
void createSamplers(std::vector<Sampler2D>& samplers, const std::vector<jp::img_stat_t>& probs, int imageWidth, int imageHeight);
inline bool samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, const cv::Point2f& pt2D, const std::vector<jp::img_center_t>& vertexs);
std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt);
inline float point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p);
inline void countInliers2D(TransHyp& hyp, const std::vector<jp::img_center_t>& vertexs, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int pixelBatch);
inline void updateHyp2D(TransHyp& hyp, int maxPixels);
inline void filterInliers2D(TransHyp& hyp, int maxInliers);
void estimateCenter(const float* probability, const float* vertmap, int batch, int height, int width, int num_classes, std::vector<jp::coord6_t>& outputs);
inline cv::Point2f getMode2D(jp::id_t objID, const cv::Point2f& pt, const std::vector<jp::img_center_t>& vertexs);

template <typename Device, typename T>
class HoughvotingOp : public OpKernel {
 public:
  explicit HoughvotingOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  // bottom_prob: (batch_size, height, width, num_classes)
  // bottom_vertex: (batch_size, height, width, 2 * num_classes)
  // top_box: (num, 6) i.e., batch_index, cls, x1, y1, x2, y2
  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_prob = context->input(0);
    const Tensor& bottom_vertex = context->input(1);

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_prob.dims() == 4,
                errors::InvalidArgument("prob must be 4-dimensional"));

    OP_REQUIRES(context, bottom_vertex.dims() == 4,
                errors::InvalidArgument("vertex must be 4-dimensional"));

    // batch size
    int batch_size = bottom_prob.dim_size(0);
    // height
    int height = bottom_prob.dim_size(1);
    // width
    int width = bottom_prob.dim_size(2);
    // num of classes
    int num_classes = bottom_prob.dim_size(3);

    // for each image, run hough voting
    std::vector<jp::coord6_t> outputs;
    for (int n = 0; n < batch_size; n++)
    {
      const float* probability = bottom_prob.flat<float>().data() + n * height * width * num_classes;
      const float* vertmap = bottom_vertex.flat<float>().data() + n * height * width * 2 * num_classes;
      estimateCenter(probability, vertmap, n, height, width, num_classes, outputs);
    }

    // Create output tensors
    // top_label
    int dims[2];
    dims[0] = outputs.size();
    dims[1] = 6;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 2, &output_shape);

    Tensor* top_box_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_box_tensor));
    float* top_box = top_box_tensor->template flat<float>().data();
    
    for(int n = 0; n < outputs.size(); n++)
    {
      jp::coord6_t box = outputs[n];
      for (int i = 0; i < 6; i++)
        top_box[n * 6 + i] = box(i);
    }
  }
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
    const Tensor& bottom_prob = context->input(0);
    const Tensor& bottom_vertex = context->input(1);

    // data should have 5 dimensions.
    OP_REQUIRES(context, bottom_prob.dims() == 4,
                errors::InvalidArgument("prob must be 4-dimensional"));

    OP_REQUIRES(context, bottom_vertex.dims() == 4,
                errors::InvalidArgument("vertex must be 4-dimensional"));

    // batch size
    int batch_size = bottom_prob.dim_size(0);
    // height
    int height = bottom_prob.dim_size(1);
    // width
    int width = bottom_prob.dim_size(2);
    // num of classes
    int num_classes = bottom_prob.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_prob.shape();
    Tensor* top_prob_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_prob_tensor));
    T* top_prob = top_prob_tensor->template flat<T>().data();
    memset(top_prob, 0, batch_size * height * width * num_classes *sizeof(T));

    TensorShape output_shape_1 = bottom_vertex.shape();
    Tensor* top_vertex_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_1, &top_vertex_tensor));
    T* top_vertex = top_vertex_tensor->template flat<T>().data();
    memset(top_vertex, 0, batch_size * height * width * 2 * num_classes *sizeof(T));
  }
};

REGISTER_KERNEL_BUILDER(Name("HoughvotingGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), HoughvotingGradOp<CPUDevice, float>);


// get probs
void getProbs(const float* probability, std::vector<jp::img_stat_t>& probs, int width, int height, int num_classes)
{
  // for each object
  for (int i = 1; i < num_classes; i++)
  {
    jp::img_stat_t img(height, width);

    #pragma omp parallel for
    for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++)
    {
      int offset = i + num_classes * (y * width + x);
      img(y, x) = probability[offset];
    }

    probs.push_back(img);
  }
}


// get centers
void getCenters(const float* vertmap, std::vector<jp::img_center_t>& vertexs, int width, int height, int num_classes)
{
  // for each object
  for (int i = 1; i < num_classes; i++)
  {
    jp::img_center_t img(height, width);

    #pragma omp parallel for
    for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++)
    {
      int channel = 2 * i;
      int offset = channel + 2 * num_classes * (y * width + x);

      jp::coord2_t obj;
      obj(0) = vertmap[offset];
      obj(1) = vertmap[offset + 1];

      img(y, x) = obj;
    }

    vertexs.push_back(img);
  }
}


// get label lists
void getLabels(const float* probability, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea)
{
  for(int i = 0; i < num_classes; i++)
    labels.push_back( std::vector<int>() );

  // for each pixel
  for(int x = 0; x < width; x++)
  for(int y = 0; y < height; y++)
  {
    float prob = -1;
    int label = -1;
    for (int i = 0; i < num_classes; i++)
    {
      int offset = i + num_classes * (y * width + x);
      if (probability[offset] > prob)
      {
        prob = probability[offset];
        label = i;
      }
    }

    labels[label].push_back(y * width + x);
  }

  for(int i = 1; i < num_classes; i++)
  {
    if (labels[i].size() > minArea)
    {
      object_ids.push_back(i);
      std::cout << "class " << i << ", " << labels[i].size() << " pixels" << std::endl;
    }
  }
}


/**
 * @brief Creates a list of samplers that return pixel positions according to probability maps.
 * 
 * This method generates numberOfObjects+1 samplers. The first sampler is a sampler 
 * for accumulated object probabilities. It samples pixel positions according to the 
 * probability of the pixel being any object (1-backgroundProbability). The 
 * following samplers correspond to the probabilities for individual objects.
 * 
 * @param samplers Output parameter. List of samplers.
 * @param probs Probability maps according to which should be sampled. One per object. The accumulated probability will be calculated in this method.
 * @param imageWidth Width of input images.
 * @param imageHeight Height of input images.
 * @return void
*/
void createSamplers(std::vector<Sampler2D>& samplers, const std::vector<jp::img_stat_t>& probs, int imageWidth, int imageHeight)
{	
  samplers.clear();
  jp::img_stat_t objProb = jp::img_stat_t::zeros(imageHeight, imageWidth);
	
  // calculate accumulated probability (any object vs background)
  #pragma omp parallel for
  for(unsigned x = 0; x < objProb.cols; x++)
  for(unsigned y = 0; y < objProb.rows; y++)
  for(auto prob : probs)
    objProb(y, x) += prob(y, x);
	
  // create samplers
  samplers.push_back(Sampler2D(objProb));
  for(auto prob : probs)
    samplers.push_back(Sampler2D(prob));
}


inline cv::Point2f getMode2D(jp::id_t objID, const cv::Point2f& pt, const std::vector<jp::img_center_t>& vertexs)
{
  jp::coord2_t mode = vertexs[objID-1](pt.y, pt.x);
  return cv::Point2f(mode(0), mode(1));
}


inline bool samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, const cv::Point2f& pt2D, const std::vector<jp::img_center_t>& vertexs)
{
  cv::Point2f obj = getMode2D(objID, pt2D, vertexs); // read out object coordinate

  eyePts.push_back(pt2D);
  objPts.push_back(obj);

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
std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
{
  std::vector<TransHyp*> workingQueue;
      
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
      workingQueue.push_back(&(it->second[h]));

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


inline void countInliers2D(TransHyp& hyp, const std::vector<jp::img_center_t>& vertexs, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int pixelBatch)
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
    cv::Point2d obj = getMode2D(hyp.objID, pt2D, vertexs);

    // inlier check
    if(point2line(hyp.center, obj, pt2D) < inlierThreshold)
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


void estimateCenter(const float* probability, const float* vertmap, int batch, int height, int width, int num_classes, std::vector<jp::coord6_t>& outputs)
{
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "num classes: " << num_classes << std::endl;

  // probs
  std::vector<jp::img_stat_t> probs;
  getProbs(probability, probs, width, height, num_classes);
  std::cout << "read probability done" << std::endl;

  // vertexs
  std::vector<jp::img_center_t> vertexs;
  getCenters(vertmap, vertexs, width, height, num_classes);
  std::cout << "read centermap done" << std::endl;
      
  //set parameters, see documentation of GlobalProperties
  int maxIterations = 10000000;
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)
  float inlierThreshold3D = 0.5;
  int ransacIterations = 256;  // 256
  int preemptiveBatch = 1000;  // 1000
  int maxPixels = 2000;  // 1000
  int refIt = 8;  // 8

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(probability, labels, object_ids, width, height, num_classes, minArea);
  std::cout << "read labels done" << std::endl;
	
  int imageWidth = width;
  int imageHeight = height;

  // create samplers for choosing pixel positions according to probability maps
  std::vector<Sampler2D> samplers;
  createSamplers(samplers, probs, imageWidth, imageHeight);
  std::cout << "created samplers: " << samplers.size() << std::endl;
		
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
	    
    // sample first point and choose object ID
    jp::id_t objID = object_ids[irand(0, object_ids.size())];

    if(objID == 0) continue;

    int pindex = irand(0, labels[objID].size());
    int index = labels[objID][pindex];
    cv::Point2f pt1(index % width, index / width);
    
    // sample first correspondence
    if(!samplePoint2D(objID, eyePts, objPts, pt1, vertexs))
      continue;

    // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt2(index % width, index / width);

    if(!samplePoint2D(objID, eyePts, objPts, pt2, vertexs))
      continue;

    // reconstruct camera
    std::vector<std::pair<cv::Point2d, cv::Point2d>> pts2D;
    for(unsigned j = 0; j < eyePts.size(); j++)
    {
      pts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(
      cv::Point2d(objPts[j].x, objPts[j].y),
      cv::Point2d(eyePts[j].x, eyePts[j].y)
      ));
    }

    Hypothesis trans(pts2D);

    // center
    cv::Point2d center = trans.getCenter();
    
    // create a hypothesis object to store meta data
    TransHyp hyp(objID, center);
    
    #pragma omp critical
    {
      hypMap[objID].push_back(hyp);
    }

    break;
  }

  // create a list of all objects where hypptheses have been found
  std::vector<jp::id_t> objList;
  std::cout << std::endl;
  for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
  {
    std::cout << "Object " << (int) hypPair.first << ": " << hypPair.second.size() << std::endl;
    objList.push_back(hypPair.first);
  }
  std::cout << std::endl;

  // create a working queue of all hypotheses to process
  std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
  // main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
  while(!workingQueue.empty())
  {
    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
      countInliers2D(*(workingQueue[h]), vertexs, labels, inlierThreshold3D, width, preemptiveBatch);
	    	    
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
    workingQueue = getWorkingQueue(hypMap, refIt);
	    
    // refine
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
    {
      updateHyp2D(*(workingQueue[h]), maxPixels);
      workingQueue[h]->refSteps++;
    }
    
    workingQueue = getWorkingQueue(hypMap, refIt);
  }

  std::cout << std::endl << "---------------------------------------------------" << std::endl;
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
  {
    std::cout << "Estimated Hypothesis for Object " << (int) it->second[h].objID << ":" << std::endl;

    cv::Point2d center = it->second[h].center;
    it->second[h].compute_width_height();

    jp::coord6_t roi;
    roi(0) = batch;
    roi(1) = it->second[h].objID;
    roi(2) = std::max(center.x - it->second[h].width_ / 2, 0.0);
    roi(3) = std::max(center.y - it->second[h].height_ / 2, 0.0);
    roi(4) = std::min(center.x + it->second[h].width_ / 2, double(width));
    roi(5) = std::min(center.y + it->second[h].height_ / 2, double(height));

    outputs.push_back(roi);
    
    std::cout << "Inliers: " << it->second[h].inliers;
    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
    std::cout << "Center " << center << std::endl;
    std::cout << "Width: " << it->second[h].width_ << " Height: " << it->second[h].height_ << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
  }
  std::cout << std::endl;
  std::cout << outputs.size() << " objects detected" << std::endl;
}
