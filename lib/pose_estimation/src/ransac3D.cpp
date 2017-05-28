/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "ransac3D.h"

using namespace jp;

Ransac3D::Ransac3D()
{
}

/**
 * @brief Thin out the inlier correspondences of the given hypothesis if there are too many. For runtime speed.
 * 
 * @param hyp Output parameter. Inlier correspondences stored in this hypothesis will the filtered.
 * @param maxInliers Maximal number of inlier correspondences to keep. Method does nothing if correspondences are fewer to begin with.
 * @return void
*/
inline void Ransac3D::filterInliers(TransHyp& hyp, int maxInliers)
{
  if(hyp.inlierPts.size() < maxInliers) return; // maximum number not reached, do nothing
      		
  std::vector<std::pair<cv::Point3d, cv::Point3d>> inlierPts; // filtered list of inlier correspondences
	
  // select random correspondences to keep
  for(unsigned i = 0; i < maxInliers; i++)
  {
    int idx = irand(0, hyp.inlierPts.size());
	    
    inlierPts.push_back(hyp.inlierPts[idx]);
  }
	
  hyp.inlierPts = inlierPts;
}


inline void Ransac3D::filterInliers2D(TransHyp& hyp, int maxInliers)
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

/**
 * @brief Recalculate the pose.
 * 
 * The hypothesis pose is recalculated using the associated inlier correspondences. The 2D bounding box of the hypothesis is also updated.
 * 
 * @param hyp Pose hypothesis to update.
 * @param camMat Camera matrix used to determine the new 2D bounding box.
 * @param imgWidth Width of the image (used for clamping the 2D bb).
 * @param imgHeight Height of the image (used for clamping the 2D bb).
 * @param bb3D 3D bounding box of the object associated with the hypothesis (used for determining the 2D bb).
 * @param maxPixels The maximal number of correspondences that should be used for recalculating the pose (for speed up).
 * @return void
*/
inline void Ransac3D::updateHyp3D(TransHyp& hyp, const cv::Mat& camMat, int imgWidth, int imgHeight, const std::vector<cv::Point3f>& bb3D, int maxPixels)
{
  if(hyp.inlierPts.size() < 4) return;
  filterInliers(hyp, maxPixels); // limit the number of correspondences
      
  // data conversion
  jp::jp_trans_t pose = jp::cv2our(hyp.pose);
  Hypothesis trans(pose.first, pose.second);	
	
  // recalculate pose
  trans.refine(hyp.inlierPts);
  hyp.pose = jp::our2cv(jp::jp_trans_t(trans.getRotation(), trans.getTranslation()));
	
  // update 2D bounding box
  hyp.bb = getBB2D(imgWidth, imgHeight, bb3D, camMat, hyp.pose);
}


inline void Ransac3D::updateHyp2D(TransHyp& hyp, int maxPixels)
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
void Ransac3D::createSamplers(std::vector<Sampler2D>& samplers, const std::vector<jp::img_stat_t>& probs, int imageWidth, int imageHeight)
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
    
/**
 * @brief Given a pixel position draw an object ID given the object probability distribution of that pixel.
 * 
 * @param pt Query pixel position.
 * @param probs Probability maps. One per object.
 * @return jp::id_t Chosen object ID.
*/
inline jp::id_t Ransac3D::drawObjID(const cv::Point2f& pt, const std::vector<jp::img_stat_t>& probs)
{
  // create a map of accumulated object probabilities at the given pixel
  std::map<float, jp::id_t> cumProb; //map of accumulated probability -> object ID
  float probCur, probSum = 0;

  for(unsigned short idx = 0; idx < probs.size(); idx++)
  {
    probCur = probs[idx](pt.y, pt.x);

    if(probCur < FLT_EPSILON) // discard probabilities close to zero
      continue;
	    
    probSum += probCur;
    cumProb[probSum] = idx + 1;
  }
	
  // choose an object based on the accumulated probability
  return cumProb.upper_bound(drand(0, probSum))->second;
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
std::vector<TransHyp*> Ransac3D::getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
{
  std::vector<TransHyp*> workingQueue;
      
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
      workingQueue.push_back(&(it->second[h]));

  return workingQueue;
}


// get 3D points from depth
void Ransac3D::getEye(unsigned char* rawdepth, jp::img_coord_t& img, jp::img_depth_t& img_depth, int width, int height, float fx, float fy, float px, float py, float depth_factor)
{
  ushort* depth = reinterpret_cast<ushort *>(rawdepth);

  img = jp::img_coord_t(height, width);
  img_depth = jp::img_depth_t(height, width);
	    
  #pragma omp parallel for
  for(int x = 0; x < width; x++)
  for(int y = 0; y < height; y++)
  {
    img(y, x) = pxToEye(x, y, depth[y * width + x], fx, fy, px, py, depth_factor);
    img_depth(y, x) = depth[y * width + x];
  }
}

jp::coord3_t Ransac3D::pxToEye(int x, int y, jp::depth_t depth, float fx, float fy, float px, float py, float depth_factor)
{
  jp::coord3_t eye;

  if(depth == 0) // depth hole -> no camera coordinate
  {
    eye(0) = 0;
    eye(1) = 0;
    eye(2) = 0;
    return eye;
  }
	
  eye(0) = (x - px) * depth / fx / depth_factor;
  eye(1) = (y - py) * depth / fy / depth_factor;
  eye(2) = depth / depth_factor;
	
  return eye;
}


// get probs
void Ransac3D::getProbs(float* probability, std::vector<jp::img_stat_t>& probs, int width, int height, int num_classes)
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


// get vertexs
void Ransac3D::getVertexs(float* vertmap, std::vector<jp::img_coord_t>& vertexs, int width, int height, int num_classes)
{
  // for each object
  for (int i = 1; i < num_classes; i++)
  {
    jp::img_coord_t img(height, width);

    #pragma omp parallel for
    for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++)
    {
      int channel = 3 * i;
      int offset = channel + 3 * num_classes * (y * width + x);

      jp::coord3_t obj;
      obj(0) = vertmap[offset];
      obj(1) = vertmap[offset + 1];
      obj(2) = vertmap[offset + 2];

      img(y, x) = obj;
    }

    vertexs.push_back(img);
  }
}


// get centers
void Ransac3D::getCenters(float* vertmap, std::vector<jp::img_center_t>& vertexs, int width, int height, int num_classes)
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
void Ransac3D::getLabels(float* probability, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea)
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


// get bb3Ds
void Ransac3D::getBb3Ds(float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
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
    
    
/**
 * @brief Main pose estimation function. Given a forest prediction it estimates poses of all objects.
 * 
 * Poses are stored in the poses member of this class.
 * 
 * @param eyeData Camera coordinate image (point cloud) generated from the depth channel.
 * @param probs Probability map for each object.
 * @param vertexs Vertex map for each object.
 * @param bb3Ds List of 3D object bounding boxes. One per object.
 * @return float Time the pose estimation took in ms.
*/
float Ransac3D::estimatePose(
	unsigned char* rawdepth,
        float* probability, float* vertmap, float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float depth_factor, float* output)
{
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "num classes: " << num_classes << std::endl;
  std::cout << "fx: " << fx << std::endl;
  std::cout << "fy: " << fy << std::endl;
  std::cout << "px: " << px << std::endl;
  std::cout << "py: " << py << std::endl;
  std::cout << "factor: " << depth_factor << std::endl;

  // extract camera coordinate image (point cloud) from depth channel
  jp::img_coord_t eyeData;
  jp::img_depth_t img_depth;
  getEye(rawdepth, eyeData, img_depth, width, height, fx, fy, px, py, depth_factor);
  std::cout << "read depth done" << std::endl;

  // probs
  std::vector<jp::img_stat_t> probs;
  getProbs(probability, probs, width, height, num_classes);
  std::cout << "read probability done" << std::endl;

  // vertexs
  std::vector<jp::img_coord_t> vertexs;
  getVertexs(vertmap, vertexs, width, height, num_classes);
  std::cout << "read vertmap done" << std::endl;

  // bb3Ds
  std::vector<std::vector<cv::Point3f>> bb3Ds;
  getBb3Ds(extents, bb3Ds, num_classes);
  std::cout << "read boxes done" << std::endl;

  GlobalProperties* gp = GlobalProperties::getInstance(); 
      
  //set parameters, see documentation of GlobalProperties
  int maxIterations = gp->tP.ransacMaxDraws;
  float minDist3D = 0.01; // in m, initial coordinates (camera and object, respectively) sampled to generate a hypothesis should be at least this far apart (for stability)
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(probability, labels, object_ids, width, height, num_classes, minArea);
  std::cout << "read labels done" << std::endl;
		
  float inlierThreshold3D = 0.01;
  int ransacIterations = gp->tP.ransacIterations;  // 256
  int preemptiveBatch = gp->tP.ransacBatchSize;  // 1000
  int maxPixels = gp->tP.ransacMaxInliers;  // 1000
  int minPixels = gp->tP.ransacMinInliers;  // 10
  int refIt = gp->tP.ransacCoarseRefinementIterations;  // 8
	
  int imageWidth = width;
  int imageHeight = height;

  cv::Mat camMat = gp->getCamMat();
  camMat.at<float>(0, 0) = fx;	
  camMat.at<float>(1, 1) = fy;
  camMat.at<float>(0, 2) = px;
  camMat.at<float>(1, 2) = py;
  std::cout << "read camera matrix:\n" << camMat << std::endl;

  // create samplers for choosing pixel positions according to probability maps
  std::vector<Sampler2D> samplers;
  createSamplers(samplers, probs, imageWidth, imageHeight);
  std::cout << "created samplers: " << samplers.size() << std::endl;
		
  // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
  std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
  float ransacTime = 0;
  StopWatch stopWatch;
	
  // sample initial pose hypotheses
  // #pragma omp parallel for
  for(unsigned h = 0; h < ransacIterations; h++)
  for(unsigned i = 0; i < maxIterations; i++)
  {
    // camera coordinate - object coordinate correspondences
    std::vector<cv::Point3f> eyePts;
    std::vector<cv::Point3f> objPts;
	  
    cv::Rect bb2D(0, 0, imageWidth, imageHeight); // initialize 2D bounding box to be the full image
	    
    // sample first point and choose object ID
    // cv::Point2f pt1 = samplers[0].drawInRect(bb2D);
    // jp::id_t objID = drawObjID(pt1, probs);
    jp::id_t objID = object_ids[irand(0, object_ids.size())];
    int pindex = irand(0, labels[objID].size());
    int index = labels[objID][pindex];
    cv::Point2f pt1(index % width, index / width);

    if(objID == 0) continue;
    
    // sample first correspondence
    if(!samplePoint(objID, eyePts, objPts, pt1, vertexs, eyeData, minDist3D))
      continue;
    
    // set a sensible search radius for other correspondences and update 2D bounding box accordingly
    float searchRadius = (fx * getMaxDist(bb3Ds[objID-1], objPts[0]) / eyePts[0].z) / 2;

    int minX = clamp(pt1.x - searchRadius, 0, imageWidth - 1);
    int maxX = clamp(pt1.x + searchRadius, 0, imageWidth - 1);
    int minY = clamp(pt1.y - searchRadius, 0, imageHeight - 1);
    int maxY = clamp(pt1.y + searchRadius, 0, imageHeight - 1);

    bb2D = cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));

    // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt2(index % width, index / width);
    // samplers[objID].drawInRect(bb2D)
    if(!samplePoint(objID, eyePts, objPts, pt2, vertexs, eyeData, minDist3D))
      continue;
    
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt3(index % width, index / width);
    if(!samplePoint(objID, eyePts, objPts, pt3, vertexs, eyeData, minDist3D))
      continue;

    // reconstruct camera
    std::vector<std::pair<cv::Point3d, cv::Point3d>> pts3D;
    for(unsigned j = 0; j < eyePts.size(); j++)
    {
      pts3D.push_back(std::pair<cv::Point3d, cv::Point3d>(
      cv::Point3d(objPts[j].x, objPts[j].y, objPts[j].z),
      cv::Point3d(eyePts[j].x, eyePts[j].y, eyePts[j].z)
      ));
    }

    Hypothesis trans(pts3D);

    // check reconstruction, sampled points should be reconstructed perfectly
    bool foundOutlier = false;
    for(unsigned j = 0; j < pts3D.size(); j++)
    {
      if(cv::norm(pts3D[j].second - trans.transform(pts3D[j].first)) < inlierThreshold3D) continue;
      foundOutlier = true;
      break;
    }
    if(foundOutlier) continue;


    // pose conversion
    jp::jp_trans_t pose;
    pose.first = trans.getRotation();
    pose.second = trans.getTranslation();
    
    // create a hypothesis object to store meta data
    TransHyp hyp(objID, jp::our2cv(pose));
    
    // update 2D bounding box
    hyp.bb = getBB2D(imageWidth, imageHeight, bb3Ds[objID-1], camMat, hyp.pose);

    //check if bounding box collapses
    if(hyp.bb.area() < minArea)
      continue;	    
    
    #pragma omp critical
    {
      hypMap[objID].push_back(hyp);
    }

    break;
  }
	
  ransacTime += stopWatch.stop();
  std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;

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
      countInliers3D(*(workingQueue[h]), vertexs, eyeData, inlierThreshold3D, minArea, preemptiveBatch);
	    	    
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
      updateHyp3D(*(workingQueue[h]), camMat, imageWidth, imageHeight, bb3Ds[workingQueue[h]->objID-1], maxPixels);
      workingQueue[h]->refSteps++;
    }
    
    workingQueue = getWorkingQueue(hypMap, refIt);
  }

  ransacTime += stopWatch.stop();
  std::cout << "Time after preemptive RANSAC: " << ransacTime << "ms." << std::endl;

  poses.clear();	

  std::cout << std::endl << "---------------------------------------------------" << std::endl;
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
  {
    std::cout << BLUETEXT("Estimated Hypothesis for Object " << (int) it->second[h].objID << ":") << std::endl;
  
    // store pose in class member
    poses[it->second[h].objID] = it->second[h];

    jp::jp_trans_t pose = jp::cv2our(it->second[h].pose);
    for(int x = 0; x < 4; x++)
    {
      for(int y = 0; y < 3; y++)
      {
        int offset = it->second[h].objID + num_classes * (y * 4 + x);
        if (x < 3)
          output[offset] = pose.first.at<double>(y, x);
        else
        {
          switch(y)
          {
            case 0: output[offset] = pose.second.x; break;
            case 1: output[offset] = pose.second.y; break;
            case 2: output[offset] = pose.second.z; break;
          }
        }
      }
    }
    
    std::cout << "Inliers: " << it->second[h].inliers;
    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
    std::cout << "Pose " << pose.first << ", " << pose.second << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
  }
  std::cout << std::endl;

  return ransacTime;
}
 
/**
 * @brief Look at a certain number of pixels and check for inliers.
 * 
 * Inliers are determined by comparing the object coordinate prediction of the random forest with the camera coordinates.
 * 
 * @param hyp Hypothesis to check.
 * @param forest Random forest that made the object coordinate prediction
 * @param leafImgs Prediction of the forest. One leaf image per tree in the forest. Each pixel stores the leaf index where the corresponding patch arrived at.
 * @param eyeData Camera coordinates of the input frame (point cloud generated from the depth channel).
 * @param inlierThreshold Allowed distance between object coordinate predictions and camera coordinates (in mm).
 * @param minArea Abort if the 2D bounding box area of the hypothesis became too small (collapses).
 * @param pixelBatch Number of pixels that should be ADDITIONALLY looked at. Number of pixels increased in each iteration by this amount.
 * @return void
*/
inline void Ransac3D::countInliers3D(
      TransHyp& hyp,
      const std::vector<jp::img_coord_t>& vertexs,
      const jp::img_coord_t& eyeData,
      float inlierThreshold,
      int minArea,
      int pixelBatch)
{
  // reset data of last RANSAC iteration
  hyp.inlierPts.clear();
  hyp.inliers = 0;

  // abort if 2D bounding box collapses
  if(hyp.bb.area() < minArea) return;

  // data conversion
  jp::jp_trans_t pose = jp::cv2our(hyp.pose);
  Hypothesis trans(pose.first, pose.second);

  hyp.effPixels = 0; // num of pixels drawn
  hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	

  int maxPt = hyp.bb.area(); // num of pixels within bounding box
  float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel

  std::mt19937 generator;
  std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept

  for(unsigned ptIdx = 0; ptIdx < maxPt;)
  {
    // convert pixel index back to x,y position
    cv::Point2f pt2D(
      hyp.bb.x + ptIdx % hyp.bb.width, 
      hyp.bb.y + ptIdx / hyp.bb.width);
    
    // skip depth holes
    if(eyeData(pt2D.y, pt2D.x)[2] == 0)
    {
      ptIdx++;
      continue;
    }
  
    // read out camera coordinate
    cv::Point3d eye(eyeData(pt2D.y, pt2D.x)[0], eyeData(pt2D.y, pt2D.x)[1], eyeData(pt2D.y, pt2D.x)[2]);
  
    hyp.effPixels++;
  
    // read out object coordinate
    cv::Point3d obj = getMode(hyp.objID, pt2D, vertexs);

    // inlier check
    if(cv::norm(eye - trans.transform(obj)) < inlierThreshold)
    {
      hyp.inlierPts.push_back(std::pair<cv::Point3d, cv::Point3d>(obj, eye)); // store object coordinate - camera coordinate correspondence
      hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
    }

    // advance to the next accepted pixel
    if(successRate < 1)
      ptIdx += std::max(1, distribution(generator));
    else
      ptIdx++;
  }
}


inline void Ransac3D::countInliers2D(
      TransHyp& hyp,
      const std::vector<jp::img_center_t>& vertexs,
      const std::vector<std::vector<int>>& labels,
      float inlierThreshold,
      int width,
      int pixelBatch)
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


inline float Ransac3D::point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p)
{
  float n1 = -n.y;
  float n2 = n.x;
  float p1 = p.x;
  float p2 = p.y;
  float x1 = x.x;
  float x2 = x.y;

  return fabs(n1 * (x1 - p1) + n2 * (x2 - p2)) / sqrt(n1 * n1 + n2 * n2);
}
  
  
/**
 * @brief  Returns the mode (center) with biggest support of the object coordiante distribution for a given pixel and tree.
 * 
 * @param objID Object for which to look up the object coordinate.
 * @param pt Pixel position to look up.
 * @param vertexs Vertex map of the objects.
 * @return cv::Point3f Center of the mode with largest support.
*/
inline cv::Point3f Ransac3D::getMode(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::img_coord_t>& vertexs)
{

  jp::coord3_t mode = vertexs[objID-1](pt.y, pt.x);
  return cv::Point3f(mode(0), mode(1), mode(2));
}

inline cv::Point2f Ransac3D::getMode2D(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::img_center_t>& vertexs)
{

  jp::coord2_t mode = vertexs[objID-1](pt.y, pt.x);
  return cv::Point2f(mode(0), mode(1));
}

/** 
 * @brief Return the minimal distance of a query point to a set of points.
 * 
 * @param pointSet Set of points.
 * @param point Query point.
 * @return double Distance.
*/
template<class T>
inline double Ransac3D::getMinDist(const std::vector<T>& pointSet, const T& point)
{
  double minDist = -1.f;
      
  for(unsigned i = 0; i < pointSet.size(); i++)
  {
    if(minDist < 0) 
      minDist = cv::norm(pointSet.at(i) - point);
    else
      minDist = std::min(minDist, cv::norm(pointSet.at(i) - point));
  }
	
  return minDist;
}
   
/** 
* @brief Return the maximal distance of a query point to a set of points.
* 
* @param pointSet Set of points.
* @param point Query point.
* @return double Distance.
*/   
template<class T>
inline double Ransac3D::getMaxDist(const std::vector<T>& pointSet, const T& point)
{
  double maxDist = -1.f;

  for(unsigned i = 0; i < pointSet.size(); i++)
  {
    if(maxDist < 0) 
      maxDist = cv::norm(pointSet.at(i) - point);
    else
      maxDist = std::max(maxDist, cv::norm(pointSet.at(i) - point));
  }

  return maxDist;
}   

/**
* @brief Returns the minimal distance of a query point to a line formed by two other points.
* 
* @param pt1 Point 1 to form the line.
* @param pt2 Point 2 to form the line.
* @param pt3 Query point.
* 
* @return double Distance.
*/
inline double Ransac3D::pointLineDistance(
  const cv::Point3f& pt1, 
  const cv::Point3f& pt2, 
  const cv::Point3f& pt3)
{
  return cv::norm((pt2 - pt1).cross(pt3 - pt1)) / cv::norm(pt2 - pt1);
}

/**
* @brief Sample a camera coordinate - object coordinate correspondence at a given pixel.
* 
* The methods checks some constraints for the new correspondence and returns false if one is violated.
* 1) There should be no depth hole at the pixel
* 2) The camera coordinate should be sufficiently far from camera coordinates sampled previously.
* 3) The object coordinate prediction should not be empty.
* 4) The object coordiante should be sufficiently far from object coordinates sampled previously.
* 
* @param objID Object for which the correspondence should be sampled for.
* @param eyePts Output parameter. List of camera coordinates. A new one will be added by this method.
* @param objPts Output parameter. List of object coordinates. A new one will be added by this method.
* @param pt2D Pixel position at which the correspondence should be sampled
* @param vertexs Vetex map of the objects.
* @param eyeData Camera coordinates of the input frame (point cloud generated from the depth channel).
* @param minDist3D The new camera coordinate should be at least this far from the previously sampled camera coordinates (in mm). Same goes for object coordinates.
* @return bool Returns true of no contraints are violated by the new correspondence.
*/
inline bool Ransac3D::samplePoint(
  jp::id_t objID,
  std::vector<cv::Point3f>& eyePts, 
  std::vector<cv::Point3f>& objPts, 
  const cv::Point2f& pt2D,
  const std::vector<jp::img_coord_t>& vertexs,
  const jp::img_coord_t& eyeData,
  float minDist3D)
{
  cv::Point3f eye(eyeData(pt2D.y, pt2D.x)[0], eyeData(pt2D.y, pt2D.x)[1], eyeData(pt2D.y, pt2D.x)[2]); // read out camera coordinate
  if(eye.z == 0) return false; // check for depth hole
  double minDist = getMinDist(eyePts, eye); // check for distance to previous camera coordinates
  if(minDist > 0 && minDist < minDist3D) return false;

  cv::Point3f obj = getMode(objID, pt2D, vertexs); // read out object coordinate
  if(obj.x == 0 && obj.y == 0 && obj.z == 0) return false; // check for empty prediction
  minDist = getMinDist(objPts, obj); // check for distance to previous object coordinates
  if(minDist > 0 && minDist < minDist3D) return false;

  eyePts.push_back(eye);
  objPts.push_back(obj);

  return true;
}


inline bool Ransac3D::samplePoint2D(
  jp::id_t objID,
  std::vector<cv::Point2f>& eyePts, 
  std::vector<cv::Point2f>& objPts, 
  const cv::Point2f& pt2D,
  const std::vector<jp::img_center_t>& vertexs)
{
  cv::Point2f obj = getMode2D(objID, pt2D, vertexs); // read out object coordinate

  eyePts.push_back(pt2D);
  objPts.push_back(obj);

  return true;
}


/**
 * @brief Main center estimation function.
 * 
*/
float Ransac3D::estimateCenter(
        float* probability, float* vertmap,
        int width, int height, int num_classes, float* output)
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

  GlobalProperties* gp = GlobalProperties::getInstance(); 
      
  //set parameters, see documentation of GlobalProperties
  int maxIterations = gp->tP.ransacMaxDraws;
  float minDist3D = 0.01; // in m, initial coordinates (camera and object, respectively) sampled to generate a hypothesis should be at least this far apart (for stability)
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(probability, labels, object_ids, width, height, num_classes, minArea);
  std::cout << "read labels done" << std::endl;
		
  float inlierThreshold3D = 0.5;
  int ransacIterations = gp->tP.ransacIterations;  // 256
  int preemptiveBatch = gp->tP.ransacBatchSize;  // 1000
  int maxPixels = 2 * gp->tP.ransacMaxInliers;  // 1000
  int minPixels = gp->tP.ransacMinInliers;  // 10
  int refIt = gp->tP.ransacCoarseRefinementIterations;  // 8
	
  int imageWidth = width;
  int imageHeight = height;

  // create samplers for choosing pixel positions according to probability maps
  std::vector<Sampler2D> samplers;
  createSamplers(samplers, probs, imageWidth, imageHeight);
  std::cout << "created samplers: " << samplers.size() << std::endl;
		
  // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
  std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
  float ransacTime = 0;
  StopWatch stopWatch;
	
  // sample initial pose hypotheses
  // #pragma omp parallel for
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
	
  ransacTime += stopWatch.stop();
  std::cout << "Time after drawing hypothesis: " << ransacTime << "ms." << std::endl;

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

  ransacTime += stopWatch.stop();
  std::cout << "Time after preemptive RANSAC: " << ransacTime << "ms." << std::endl;

  std::cout << std::endl << "---------------------------------------------------" << std::endl;
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
  {
    std::cout << BLUETEXT("Estimated Hypothesis for Object " << (int) it->second[h].objID << ":") << std::endl;

    cv::Point2d center = it->second[h].center;
    it->second[h].compute_width_height();
    int offset = 4 * it->second[h].objID;
    output[offset] = center.x;
    output[offset+1] = center.y;
    output[offset+2] = it->second[h].width_;
    output[offset+3] = it->second[h].height_;
    
    std::cout << "Inliers: " << it->second[h].inliers;
    std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
    std::cout << "Refined " << it->second[h].refSteps << " times. " << std::endl;
    std::cout << "Center " << center << std::endl;
    std::cout << "Width: " << it->second[h].width_ << " Height: " << it->second[h].height_ << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
  }
  std::cout << std::endl;

  return ransacTime;
}
