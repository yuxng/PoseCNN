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

#include "opencv2/opencv.hpp"

#define EPS 0.00000001
#define PI 3.1415926

/** Several important types used troughout all this code. If types have to be changed, it can be done here, conveniently. */

namespace jp
{
    // object IDs
    typedef unsigned short id_t;

    // object coordinates 
    typedef float coord1_t; // one dimension
    typedef cv::Vec<coord1_t, 2> coord2_t; // two dimensions
    typedef cv::Vec<coord1_t, 3> coord3_t; // three dimensions

    // label types
    typedef unsigned short cell_t; // quantized coordinates
    typedef unsigned short label_t; // labels of joint distribution of objects and cells
    typedef std::vector<unsigned int> histogram_t; // histograms of labels

    // rgb-d
    typedef cv::Vec<uchar, 3> bgr_t;
    typedef unsigned short depth_t;

    // image types
    typedef cv::Mat_<coord3_t> img_coord_t; // object coodinate images
    typedef cv::Mat_<coord2_t> img_center_t; // center coodinate images
    typedef cv::Mat_<bgr_t> img_bgr_t; // color images
    typedef cv::Mat_<depth_t> img_depth_t; // depth images
    typedef cv::Mat_<label_t> img_label_t; // label images (quantized object coordinates + object ID)
    typedef cv::Mat_<id_t> img_id_t; // segmentation images

    /**
     * @brief One mode of a GMM.
     */
    struct mode_t
    {
     
	mode_t()
	{
	    mean = jp::coord3_t(0, 0, 0);
	    covar = cv::Mat::zeros(3, 3, CV_32F);
	    support = 0;
	}
      
	jp::coord3_t mean; // mean of the mode
	cv::Mat_<float> covar; // 3x3 covariance of the mode
	unsigned support; // number of samples that belong to this mode
	
	/**
	 * @brief Compare modes by their support size.
	 * 
	 * @return bool
	 */
	bool operator<(const mode_t& mode) const
	{ 
	    return (this->support < mode.support);
	}
    };  
    
    typedef cv::Mat_<size_t> img_leaf_t; // image of leaf indices per pixel
    typedef cv::Mat_<float> img_stat_t; // image containing some statistic per pixel, e.g. an object probability
    

    /**
     * @brief RGB-D image. 
     */
    struct img_bgrd_t
    {
	img_bgr_t bgr;
	img_depth_t depth;
    };

    /**
     * @brief Container type for input data the auto context forest works on.
     */
    struct img_data_t
    {
	img_id_t seg; // object segmentation
	img_bgr_t colorData; // color image
	
	// auto-context output of the previous forest
	std::vector<img_label_t> labelData; // previous segmentation prediction, one image per object
	std::vector<img_coord_t> coordData; // previous object coordinate prediction, one image per object
    };
    
    /**
     * @brief Ground truth information about objects, given per image.
     */
    struct info_t
    {
	std::string name; // name of the object
	cv::Mat_<float> rotation; // ground truth 3x3 rotation matrix
	cv::Vec<float, 3> center; // ground truth translation im meters
	cv::Vec<float, 3> extent; // object extent (width, length, height) given in meters
	bool visible; // is the object visible in this image?
	float occlusion; // percentage of the object area occluded in this image
	
	/**
	 * @brief Constructor.
	 * 
	 * @param v Is the object visible? Default is true.
	 */
	info_t(bool v = true)
	{
	    rotation = cv::Mat_<float>::eye(3, 3);
	    center = cv::Vec<float, 3>(0, 0, -1);
	    extent = cv::Vec<float, 3>(1, 1, 1);
	    visible = v;
	    occlusion = 0;
	}
    };  
    
    typedef std::pair<cv::Mat, cv::Mat> cv_trans_t; // object (or camera) pose as expected by OpenCV methods
    typedef std::pair<cv::Mat, cv::Point3d> jp_trans_t; // object (or camera) as expected by some of our methods and types
    
    /**
     * @brief Convert an OpenCV float matrix to a double matrix.
     * 
     * @param fmat Float matrix.
     * @return cv::Mat Double matrix.
     */
    inline cv::Mat float2double(cv::Mat& fmat) 
    {
	cv::Mat_<double> dmat(fmat.rows, fmat.cols);
	
	for(unsigned i = 0; i < fmat.cols; i++)
	for(unsigned j = 0; j < fmat.rows; j++)
	    dmat(j, i) = fmat.at<float>(j, i);
	
	return dmat;
    }
    
    /**
     * @brief Convert an OpenCV double matrix to a float matrix.
     * 
     * @param dmat Double matrix.
     * @return cv::Mat Float matrix.
     */
    inline cv::Mat double2float(cv::Mat& dmat) 
    {
	cv::Mat_<float> fmat(dmat.rows, dmat.cols);
	
	for(unsigned i = 0; i < dmat.cols; i++)
	for(unsigned j = 0; j < dmat.rows; j++)
	    fmat(j, i) = dmat.at<double>(j, i);
	
	return fmat;
    }     
    
    
    /**
     * @brief Convert a pose in our custom format to the OpenCV format.
     * 
     * A change of the coordinate frame takes place because of different conventions.
     * 
     * @param trans Pose in our custom format.
     * @return jp::cv_trans_t Pose in OpenCV format.
     */
    inline cv_trans_t our2cv(const jp_trans_t& trans)
    {
	// map opencv coordinate system to ours (180deg rotation around x)
	cv::Mat rmat = trans.first.clone(), rvec;
	cv::Rodrigues(rmat, rvec);
	
	cv::Mat tvec(3, 1, CV_64F);
	tvec.at<double>(0, 0) = trans.second.x;
	tvec.at<double>(1, 0) = trans.second.y;
	tvec.at<double>(2, 0) = trans.second.z;

	return cv_trans_t(rvec, tvec);
    }

    
    /**
     * @brief Convert a pose in the OpenCV format to our custom pose format.
     * 
     * A change of the coordinate frame takes place because of different conventions.
     * 
     * @param trans Pose in OpenCV format.
     * @return jp::jp_trans_t Pose in our custom format.
     */
    inline jp_trans_t cv2our(const cv_trans_t& trans)
    {
	// map data types
	cv::Mat rmat;
	cv::Rodrigues(trans.first, rmat);
	cv::Point3d tpt(trans.second.at<double>(0, 0), trans.second.at<double>(1, 0), trans.second.at<double>(2, 0));

	// result may be reconstructed behind the camera
	if(cv::determinant(rmat) < 0)
	{
	    tpt = -tpt;
	    rmat = -rmat;
	}
	
	return jp_trans_t(rmat, tpt);      
    }
}
