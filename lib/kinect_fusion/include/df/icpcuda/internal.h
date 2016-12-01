/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef INTERNAL_HPP_
#define INTERNAL_HPP_

#include "device_array.hpp"

#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <Eigen/Eigen>

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

/** \brief Camera intrinsics structure
  */
struct Intr
{
    float fx, fy, cx, cy;
    Intr () : fx(0), fy(0), cx(0), cy(0) {}
    Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    Intr operator()(int level_index) const
    {
        int div = 1 << level_index;
        return (Intr (fx / div, fy / div, cx / div, cy / div));
    }
};

void estimateStep(const Eigen::Matrix<float,3,3,Eigen::DontAlign> & R_prev_curr,
                  const Eigen::Matrix<float,3,1,Eigen::DontAlign> & t_prev_curr,
                  const DeviceArray2D<float>& vmap_curr,
                  const DeviceArray2D<float>& nmap_curr,
                  const Intr& intr,
                  const DeviceArray2D<float>& vmap_prev,
                  const DeviceArray2D<float>& nmap_prev,
                  float dist_thresh,
                  float angle_thresh,
                  DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> & sum,
                  DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> & out,
                  float * matrixA_host,
                  float * vectorB_host,
                  float * residual_inliers,
                  int threads,
                  int blocks);

void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst);

void createVMap(const Intr& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & vmap, const float depthCutoff);
void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap);

#endif /* INTERNAL_HPP_ */
