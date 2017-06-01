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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include "types.h"


/**
 * @brief Class that holds a pose hypothesis. 
 * 
 * Holds various methods for calculating poses, refinement, conversion of formats and transfromation.
 * 
 */
class Hypothesis{
public:
	

        /**
         * @brief Creates and identity transformation.
         */
        Hypothesis();

        Hypothesis(cv::Point2d center);
	
	/**
	 * @brief Create a pose from a rotation matrix and a translation vector.
	 * 
	 * @param rot 3x3 double rotation matrix.
	 * @param trans Translation vector.
	 */
	Hypothesis(cv::Mat rot,cv::Point3d trans);
	
	/**
	 * @brief Creates a pose from a 4x4 transformation matrix.
	 * 
	 * @param transform 4x4 double transformation matrix: Upper 3x3 matrix is the rotation matrix, last column is the translation vector.
	 */
	Hypothesis(cv::Mat transform);
	
	/**
	 * @brief Estimates a poses from 3D-3D point correspondences.
	 * 
	 * Uses the Kabsch algorithms, i.e. calculates the poses that minimizes the squared distances of correspondences.
	 * 
	 * @param points Correspondences.
	 */
	Hypothesis(std::vector<std::pair<cv::Point3d,cv::Point3d>> points);

        Hypothesis(std::vector<std::pair<cv::Point2d, cv::Point2d>> points);
	
	/**
	 * @brief Create a pose from a Rodrigues vector and a translation vector.
	 * 
	 * @param rodVecAndTrans 6-entry vector: first 3 entries are the rotation as Rodrigues vector, last 3 entries are the translation vector.
	 */
	Hypothesis(std::vector<double> rodVecAndTrans );
	
	/**
	 * @brief Create a pose from ground truth pose information.
	 * 
	 * @param info Ground truth pose.
	 */
	Hypothesis(jp::info_t info);
	
	/**
	 * @brief Add new 3D-3D point correspondences and recalculate the pose using Kabsch.
	 * 
	 * @param points New set of correspondences.
	 * @return void
	 */
	void refine(std::vector<std::pair<cv::Point3d,cv::Point3d>> points);
	
	/**
	 * @brief Recalculate the pose using the covariance matrix and means of point correspondences.
	 * 
	 * @param coV 3x3 double covariance matrix of point correspondences.
	 * @param pointsA Mean point of left side of correspondences.
	 * @param pointsB Mean point of right side of correspondences.
	 * @return void
	 */
	void refine(cv::Mat& coV,cv::Point3d pointsA,cv::Point3d pointsB);


        cv::Point2d calcCenter(std::vector<std::pair<cv::Point2d, cv::Point2d>> points);
	
	/**
	 * @brief Returns the translation vector.
	 * 
	 * @return cv::Point3d Translation vector.
	 */
	cv::Point3d getTranslation() const;


        cv::Point2d getCenter() const;
	
	/**
	 * @brief Returns the 3x3 double rotation matrix.
	 * 
	 * @return cv::Mat Rotation matrix.
	 */
	cv::Mat getRotation() const;
	
	/**
	 * @brief Returns the 3x3 double inverted rotation matrix.
	 * 
	 * @return cv::Mat Inverted rotation matrix.
	 */
	cv::Mat getInvRotation() const;
	
	/**
	 * @brief Returns the rotation as a 3x1 double Rodrigues vector.
	 * 
	 * @return cv::Mat Rodrigues vector.
	 */
	cv::Mat getRodriguesVector() const;
	
	/**
	 * @brief Returns the pose as a 6-entry vector: First 3 entries are the rotation as Rodriguez vector, last 3 entries are the translation vector.
	 * 
	 * @return std::vector< double, std::allocator< void > > 6-entry pose vector.
	 */
	std::vector<double> getRodVecAndTrans() const;
	
	/**
	 * @brief Set the rotation of the pose.
	 * 
	 * @param rot 3x3 double rotation matrix.
	 * @return void
	 */
	void setRotation(cv::Mat rot);
	
	
	/**
	 * @brief Set the translation vector of the pose.
	 * 
	 * @param trans Translation vector.
	 * @return void
	 */
	void setTranslation(cv::Point3d trans);
	
	
	/**
	 * @brief Get the 4x4 double transformation matrix.
	 * 
	 * @return cv::Mat Transformation matrix.
	 */
	cv::Mat getTransformation() const;
	
	/**
	 * @brief Returns a hypothesis with of the inverted pose.
	 * 
	 * @return Hypothesis Inverted pose.
	 */
	Hypothesis getInv();

	
	/**
	 * @brief Multiply this hypothesis with another one.
	 * 
	 * @param other Second hypothesis.
	 * @return Hypothesis Result.
	 */
	Hypothesis operator*(const Hypothesis& other) const;
	
	/**
	 * @brief Multiplies the inverted hypotheses.
	 * 
	 * @param other Second hypothesis.
	 * @return Hypothesis Result.
	 */
	Hypothesis operator/(const Hypothesis& other) const;
	
	
       /**
	 * @brief Transforms a 3d point from object-system into kinect-system.
	 * 
	 * @param p Point to be transformed.
	 * @param isNormal Optional. Is p a normal? Then translation is ommited.
	 * @return Point in kinect-system.
	 */
	cv::Point3d transform(cv::Point3d p, bool isNormal = false);

       /**
	 * @brief Transforms a 3d point from kinect-system into object-system.
	 * 
	 * @param p Point to be transformed.
	 * @return Point in object-system.
	 */
	cv::Point3d invTransform(cv::Point3d p);
	
	/**
	 * @brief Calculate the rotational distance (deg) to another Hypothesis.
	 * @param h Hypothesis to compute distance to.
	 * @return Distance.
	 */
	double calcAngularDistance(Hypothesis h);

	~Hypothesis();
	
	/**
	 * @brief Calculates a pose using the Kabsch algorithm.
	 * 
	 * @param coV 3x3 double covariance matrix of point correspondences.
	 * @param pointsA Mean point of left side of correspondences.
	 * @param pointsB Mean point of right side of correspondences.
	 * @return std::pair< cv::Mat, cv::Point3d > 3x3 double rotation matrix and translation vector.
	 */
	static std::pair<cv::Mat,cv::Point3d> calcRigidBodyTransform(cv::Mat& coV,cv::Point3d pointsA, cv::Point3d pointsB);
private:
	cv::Mat rotation; 
	cv::Mat invRotation;
	cv::Point3d translation;
	std::vector<std::pair<cv::Point3d,cv::Point3d>> points; // point correspondences used to calculated this pose, stored for refinement later

	cv::Point2d center;
	std::vector<std::pair<cv::Point2d,cv::Point2d>> points2D; // point correspondences used to calculated this pose, stored for refinement later
	
	/**
	 * @brief Calculates a pose from 3D-3D point correspondences using the Kabsch algorithm.
	 * 
	 * @param points Point correspondences.
	 * @return std::pair< cv::Mat, cv::Point3d > 3x3 double rotation matrix and translation vector.
	 */
	static std::pair<cv::Mat,cv::Point3d> calcRigidBodyTransform(std::vector<std::pair<cv::Point3d, cv::Point3d>> points);
};
