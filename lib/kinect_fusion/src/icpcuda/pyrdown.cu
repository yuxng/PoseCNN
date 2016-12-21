/*
 * Software License Agreement(BSD License)
 *
 *  Point Cloud Library(PCL) - www.pointclouds.org
 *  Copyright(c) 2011, Willow Garage, Inc.
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
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <df/icpcuda/internal.h>
#include <df/icpcuda/safe_call.hpp>

__global__ void pyrDownGaussKernel(const PtrStepSz<unsigned short> src, PtrStepSz<unsigned short> dst, float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr(2 * y)[2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
    int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f} ;

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            int val = src.ptr(2*y + yi)[2*x + xi];

            if(abs(val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst.ptr(y)[x] = static_cast<int>(sum /wall);
}

void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst)
{
    dst.create(src.rows() / 2, src.cols() / 2);

    dim3 block(32, 8);
    dim3 grid(divUp(dst.cols(), block.x), divUp(dst.rows(), block.y));

    const float sigma_color = 30;

    pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaSafeCall( cudaGetLastError() );
};

__global__ void computeVmapKernel(const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff, float depthFactor)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr(v)[u] / depthFactor; // load and convert: mm -> meters

        if(z != 0 && z < depthCutoff)
        {
            float vx = z *(u - cx) * fx_inv;
            float vy = z *(v - cy) * fy_inv;
            float vz = z;

            vmap.ptr(v )[u] = vx;
            vmap.ptr(v + depth.rows)[u] = vy;
            vmap.ptr(v + depth.rows * 2)[u] = vz;
        }
        else
        {
            vmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        }
    }
}

void createVMap(const Intr& intr, const DeviceArray2D<unsigned short> & depth, DeviceArray2D<float> & vmap, const float depthCutoff, const float depthFactor)
{
    vmap.create(depth.rows() * 3, depth.cols());

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(depth.cols(), block.x);
    grid.y = divUp(depth.rows(), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff, depthFactor);
    cudaSafeCall(cudaGetLastError());
}

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u >= cols || v >= rows)
        return;

    if(u == cols - 1 || v == rows - 1)
    {
        nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    Eigen::Matrix<float,3,1,Eigen::DontAlign> v00, v01, v10;
    v00(0) = vmap.ptr(v)[u];
    v01(0) = vmap.ptr(v)[u + 1];
    v10(0) = vmap.ptr(v + 1)[u];

    if(!isnan(v00(0)) && !isnan(v01(0)) && !isnan(v10(0)))
    {
        v00(1) = vmap.ptr(v + rows)[u];
        v01(1) = vmap.ptr(v + rows)[u + 1];
        v10(1) = vmap.ptr(v + 1 + rows)[u];

        v00(2) = vmap.ptr(v + 2 * rows)[u];
        v01(2) = vmap.ptr(v + 2 * rows)[u + 1];
        v10(2) = vmap.ptr(v + 1 + 2 * rows)[u];

        Eigen::Matrix<float,3,1,Eigen::DontAlign> r = (v01 - v00).cross(v10 - v00).normalized();

        nmap.ptr(v)[u] = r(0);
        nmap.ptr(v + rows)[u] = r(1);
        nmap.ptr(v + 2 * rows)[u] = r(2);
    }
    else
        nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create(vmap.rows(), vmap.cols());

    int rows = vmap.rows() / 3;
    int cols = vmap.cols();

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = divUp(cols, block.x);
    grid.y = divUp(rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall(cudaGetLastError());
}

