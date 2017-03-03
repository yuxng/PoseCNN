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
#include "thread_rand.h"
#include <array>


/**
 * @brief Class for drawing pixel positions according to weights given per pixel.
  */
class Sampler2D
{
public:
    /**
     * @brief Constructor.
     * 
     * @param probs Map of probabilities or weights per pixel according to which pixel positions should be sampled.
     */
    Sampler2D(const jp::img_stat_t& probs)
    {
	cv::integral(probs, integral);
    }

    /**
     * @brief Samples a random pixel position in the given 2D window.
     * 
     * @param bb2D 2D window the sample should lie in.
     * @return cv::Point2f Random pixel position.
     */
    cv::Point2f drawInRect(const cv::Rect& bb2D)
    {
	int minX = bb2D.tl().x;
	int maxX = bb2D.br().x - 1;
	int minY = bb2D.tl().y;
	int maxY = bb2D.br().y - 1;
      
	// choose the accumulated weight of the pixels to sample 
	double randNum = drand(0, 1) * getSum(minX, minY, maxX, maxY);
	
	// search for the pixel that statisfies the accumulated weight
	return drawInRect(minX, minY, maxX, maxY, randNum);
    }
    
public:
  
    /**
     * @brief Calclate the accumulated weights within the given bound.
     * 
     * @param minX Minimum X value of the search window.
     * @param maxX Maximum X value of the search window.
     * @param minY Minimum Y value of the search window.
     * @param maxY Maximum Y value of the search window.
     * @return double Sum of weights.
     */
    inline double getSum(int minX, int minY, int maxX, int maxY)
    {
	double sum = integral(maxY + 1, maxX + 1);
	if(minX > 0) sum -= integral(maxY + 1, minX);
	if(minY > 0) sum -= integral(minY, maxX + 1);
	if(minX > 0 && minY > 0) sum += integral(minY, minX);
	return sum;
    }
    
    /**
     * @brief Recursive search for the pixel position which has the given accumulated weight.
     * 
     * @param minX Minimum X value of the search window.
     * @param maxX Maximum X value of the search window.
     * @param minY Minimum Y value of the search window.
     * @param maxY Maximum Y value of the search window.
     * @param randNum Accumulated weight to look for.
     * @return cv::Point2f Pixel position with the given accumulated weight.
     */
    cv::Point2f drawInRect(int minX, int minY, int maxX, int maxY, double randNum)
    {
	double halfInt;
	
	// first search in X direction
	if(maxX - minX > 0)
	{
	    // binary search, does the pixel lie in the left or right half of the search window?
	    halfInt = getSum(minX, minY, (minX + maxX) / 2, maxY);
	    
	    if(randNum > halfInt) 
		return drawInRect((minX + maxX) / 2 + 1, minY, maxX, maxY, randNum - halfInt);
	    else 
		return drawInRect(minX, minY, (minX + maxX) / 2, maxY, randNum);
	}

	// search in Y direction
	if(maxY - minY > 0)
	{
	    // binary search, does the pixel lie in the upper or lower half of the search window?
	    halfInt = getSum(minX, minY, maxX, (minY + maxY) / 2);
	    
	    if(randNum > halfInt) 
		return drawInRect(minX, (minY + maxY) / 2 + 1, maxX, maxY, randNum - halfInt);
	    else 
		return drawInRect(minX, minY, minX, (minY + maxY) / 2, randNum);
	}

	return cv::Point2f(maxX, maxY);
    }    
  
    cv::Mat_<double> integral; // integral image (map of accumulated weights) used in binary search
};
