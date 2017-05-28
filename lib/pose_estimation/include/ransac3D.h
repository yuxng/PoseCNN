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

#pragma once

#include "types.h"
#include "util.h"
#include "sampler2D.h"
#include "detection.h"
#include "stop_watch.h"
#include "Hypothesis.h"

#include <nlopt.hpp>
#include <omp.h>
#include <cfloat>

namespace jp {


    /**
     * @brief Struct that bundels data that is held per pose hypothesis during optimization.
     */
    struct TransHyp
    {
	TransHyp() {}
	TransHyp(jp::id_t objID, jp::cv_trans_t pose) : pose(pose), objID(objID), inliers(0), maxPixels(0), effPixels(0), refSteps(0), likelihood(0) {}
        TransHyp(jp::id_t objID, cv::Point2d center) : center(center), objID(objID), inliers(0), maxPixels(0), effPixels(0), refSteps(0), likelihood(0) {}
      
	jp::id_t objID; // ID of the object this hypothesis belongs to
	jp::cv_trans_t pose; // the actual transformation

        cv::Point2d center; // object center
        float width_;
        float height_;
	
	cv::Rect bb; // 2D bounding box of the object under this pose hypothesis
	
	std::vector<std::pair<cv::Point3d, cv::Point3d> > inlierPts; // list of object coordinate - camera coordinate correspondences that support this hypothesis
	std::vector<std::pair<cv::Point2d, cv::Point2d> > inlierPts2D; // list of object coordinate - camera coordinate correspondences that support this hypothesis
	
	int maxPixels; // how many pixels should be maximally drawn to score this hyp
	int effPixels; // how many pixels habe effectively drawn (bounded by projection size)
	
	int inliers; // how many of them were inliers
	float likelihood; // likelihood of this hypothesis (optimization using uncertainty)

	int refSteps; // how many iterations has this hyp been refined?
	
	/**
	 * @brief Returns a score for this hypothesis used to sort in preemptive RANSAC.
	 * 
	 * @return float Score.
	 */
	float getScore() const 	{ return inliers; }
	
	/**
	 * @brief Fraction of inlier pixels as determined by RANSAC.
	 * 
	 * @return float Fraction of inliers.
	 */	
	float getInlierRate() const { return inliers / (float) effPixels; }
	
	/**
	 * @brief Operator used in sorting hypothesis. Compares scores.
	 * 
	 * @return bool True if this hypothesis' score is bigger.
	 */
	bool operator < (const TransHyp& hyp) const { return (getScore() > hyp.getScore()); }

        void compute_width_height()
        {
          float w = -1;
          float h = -1;
          for(int i = 0; i < inliers; i++)
          {
            float x = fabs(inlierPts2D[i].second.x - center.x);
            float y = fabs(inlierPts2D[i].second.y - center.y);
            if (x > w)
              w = x;
            if (y > h)
              h = y;
          }
          width_ = 2 * w;
          height_ = 2 * h;
        }
    };


/**
 * @brief RANSAC class of finding poses based on object coordinate predictions in the RGB-D case.
 */
class Ransac3D
{
public:
    Ransac3D();

    inline void filterInliers(TransHyp& hyp, int maxInliers);

    inline void filterInliers2D(TransHyp& hyp, int maxInliers);

    inline void updateHyp3D(TransHyp& hyp, const cv::Mat& camMat, int imgWidth, int imgHeight, const std::vector<cv::Point3f>& bb3D, int maxPixels);

    inline void updateHyp2D(TransHyp& hyp, int maxPixels);
    
    void createSamplers(std::vector<Sampler2D>& samplers, const std::vector<jp::img_stat_t>& probs, int imageWidth, int imageHeight);
    
    inline jp::id_t drawObjID(const cv::Point2f& pt, const std::vector<jp::img_stat_t>& probs);
    
    std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt);
    
    float estimatePose(
	unsigned char* rawdepth,
        float* probability, float* vertmap, float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float depth_factor, float* output);

    float estimateCenter(
        float* probability, float* vertmap,
        int width, int height, int num_classes, float* output);

    void getEye(unsigned char* rawdepth, jp::img_coord_t& img, jp::img_depth_t& img_depth, int width, int height, float fx, float fy, float px, float py, float depth_factor);
    jp::coord3_t pxToEye(int x, int y, jp::depth_t depth, float fx, float fy, float px, float py, float depth_factor);

    void getProbs(float* probability, std::vector<jp::img_stat_t>& probs, int width, int height, int num_classes);

    void getLabels(float* probability, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea);

    void getVertexs(float* vertmap, std::vector<jp::img_coord_t>& vertexs, int width, int height, int num_classes);

    void getCenters(float* vertmap, std::vector<jp::img_center_t>& vertexs, int width, int height, int num_classes);

    void getBb3Ds(float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
    
private:
 
    inline void countInliers3D(
      TransHyp& hyp,
      const std::vector<jp::img_coord_t>& vertexs,
      const jp::img_coord_t& eyeData,
      float inlierThreshold,
      int minArea,
      int pixelBatch);

    inline void countInliers2D(
      TransHyp& hyp,
      const std::vector<jp::img_center_t>& vertexs,
      const std::vector<std::vector<int>>& labels,
      float inlierThreshold,
      int width,
      int pixelBatch);
  
    inline cv::Point3f getMode(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::img_coord_t>& vertexs);

    inline cv::Point2f getMode2D(
	jp::id_t objID,
	const cv::Point2f& pt, 
	const std::vector<jp::img_center_t>& vertexs);

    template<class T>
    inline double getMinDist(const std::vector<T>& pointSet, const T& point);
   
    template<class T>
    inline double getMaxDist(const std::vector<T>& pointSet, const T& point);  
   
    inline double pointLineDistance(
	const cv::Point3f& pt1, 
	const cv::Point3f& pt2, 
	const cv::Point3f& pt3);

    inline float point2line(cv::Point2d p, cv::Point2f n, cv::Point2f x);
    
    inline bool samplePoint(
	jp::id_t objID,
	std::vector<cv::Point3f>& eyePts, 
	std::vector<cv::Point3f>& objPts, 
	const cv::Point2f& pt2D,
	const std::vector<jp::img_coord_t>& vertexs,
	const jp::img_coord_t& eyeData,
	float minDist3D);

    inline bool samplePoint2D(
        jp::id_t objID,
        std::vector<cv::Point2f>& eyePts, 
        std::vector<cv::Point2f>& objPts, 
        const cv::Point2f& pt2D,
        const std::vector<jp::img_center_t>& vertexs);
    
public:
    std::map<jp::id_t, TransHyp> poses; // Poses that have been estimated. At most one per object. Run estimatePose to fill this member.
};

    /**
     * @brief Data used in NLOpt callback loop.
     */
    struct DataForOpt
    {
	TransHyp* hyp; // pointer to the data attached to the hypothesis being optimized.
	Ransac3D* ransac; // pointer to the RANSAC object for access of various methods.
    };

}
