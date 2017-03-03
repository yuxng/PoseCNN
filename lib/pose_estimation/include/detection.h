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
#include "properties.h"

/** Methods for calculating 2D bounding box and comparing 2D BB detections. */

/**
 * @brief Get a 8 vertex 3D bounding box from the given extent. The BB will be zero-centered.
 * 
 * @param extent Extent of the object (width, height, depth)
 * @return std::vector< cv::Point3f, std::allocator< void > > Bounding box as a list of its 8 vertices.
 */
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

/**
 * @brief Get a 2D bounding box by projecting the 3D bounding box (under a given pose) into the image.
 * 
 * In case of full scene objects (i.e. scenes) the 2D bb will always cover the full image.
 * 
 * @param imageWidth Width of input images (used for clamping).
 * @param imageHeight Height of input images (used for clamping).
 * @param bb3D 3D boudning box of the object.
 * @param camMat Camera matrix 3x3 intrinsic camera parameters.
 * @param trans Object pose.
 * @return cv::Rect 2D bounding box.
 */
inline cv::Rect getBB2D(
  int imageWidth, int imageHeight,
  const std::vector<cv::Point3f>& bb3D,
  const cv::Mat& camMat,
  const jp::cv_trans_t& trans)
{
    GlobalProperties* gp = GlobalProperties::getInstance();
    
    if(gp->fP.fullScreenObject) // for scenes the 2D bounding box is always the complete image
	return cv::Rect(0, 0, gp->fP.imageWidth, gp->fP.imageHeight);
    
    // project 3D bounding box vertices into the image
    std::vector<cv::Point2f> bb2D;
    cv::projectPoints(bb3D, trans.first, trans.second, camMat, cv::Mat(), bb2D);
    
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

inline cv::Rect getBB2D(
  int imageWidth, int imageHeight,
  const std::vector<cv::Point3f>& bb3D,
  const cv::Mat& camMat,
  const jp::jp_trans_t& trans)
{   
    // project 3D bounding box vertices into the image
    std::vector<cv::Point2f> bb2D;

    // construct the projection matrix
    cv::Mat_<float> RT(3, 4);
    trans.first.copyTo(RT(cv::Range(0,3), cv::Range(0,3)));
    RT.at<float>(0, 3) = trans.second.x;
    RT.at<float>(1, 3) = trans.second.y;
    RT.at<float>(2, 3) = trans.second.z;

    cv::Mat P = camMat * RT;

    // projection
    for (int i = 0; i < bb3D.size(); i++)
    {
      cv::Mat x3d = cv::Mat::zeros(4, 1, CV_32F);
      x3d.at<float>(0, 0) = bb3D[i].x;
      x3d.at<float>(1, 0) = bb3D[i].y;
      x3d.at<float>(2, 0) = bb3D[i].z;
      x3d.at<float>(3, 0) = 1.0;

      cv::Mat x2d = P * x3d;
      bb2D.push_back(cv::Point2f(x2d.at<float>(0, 0) / x2d.at<float>(2, 0), x2d.at<float>(1, 0) / x2d.at<float>(2, 0)));
    }
    
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

/**
 * @brief Calculates the intersection over union of two 2D bounding boxes.
 * 
 * @param bb1 Bounding box 1.
 * @param bb2 Bounding box 2.
 * @return float Intersection over union.
 */
inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2)
{
    cv::Rect intersection = bb1 & bb2;
    return (intersection.area() / (float) (bb1.area() + bb2.area() - intersection.area()));
}
