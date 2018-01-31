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
#include <nlopt.hpp>
#include <omp.h>
#include <cfloat>
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
        float x1_, y1_, x2_, y2_;
	
	cv::Rect bb; // 2D bounding box of the object under this pose hypothesis
	
	std::vector<std::pair<cv::Point3d, cv::Point3d> > inlierPts; // list of object coordinate - camera coordinate correspondences that support this hypothesis
	std::vector<std::pair<cv::Point3d, cv::Point2d> > inlierPts2D; // list of object coordinate - camera coordinate correspondences that support this hypothesis
	
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

        void compute_box()
        {
          float x1 = 100000;
          float y1 = 100000;
          float x2 = -1;
          float y2 = -1;
          for(int i = 0; i < inliers; i++)
          {
            float x = inlierPts2D[i].second.x;
            float y = inlierPts2D[i].second.y;
            if (x1 > x)
              x1 = x;
            if (x2 < x)
              x2 = x;
            if (y1 > y)
              y1 = y;
            if (y2 < y)
              y2 = y;
          }
          x1_ = x1;
          x2_ = x2;
          y1_ = y1;
          y2_ = y2;
        }
    };
