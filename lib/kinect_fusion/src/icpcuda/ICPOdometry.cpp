/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include <df/icpcuda/ICPOdometry.h>

ICPOdometry::ICPOdometry(int width,
                          int height,
                          float cx, float cy, float fx, float fy,
                          float distThresh,
                          float angleThresh)
: lastError(0),
  lastInliers(width * height),
  dist_thresh(distThresh),
  angle_thresh(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumData.create(MAX_THREADS);
    outData.create(1);

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_prev.resize(NUM_PYRS);
    nmaps_prev.resize(NUM_PYRS);

    vmaps_curr.resize(NUM_PYRS);
    nmaps_curr.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create(pyr_rows, pyr_cols);

        vmaps_prev[i].create(pyr_rows*3, pyr_cols);
        nmaps_prev[i].create(pyr_rows*3, pyr_cols);

        vmaps_curr[i].create(pyr_rows*3, pyr_cols);
        nmaps_curr[i].create(pyr_rows*3, pyr_cols);
    }
}

ICPOdometry::~ICPOdometry()
{

}

void ICPOdometry::initICP(unsigned short * depth, const float depthCutoff, const float depthFactor)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr[i], depthCutoff, depthFactor);
        createNMap(vmaps_curr[i], nmaps_curr[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::initICPModel(unsigned short * depth, const float depthCutoff, const float depthFactor)
{
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_prev[i], depthCutoff, depthFactor);
        createNMap(vmaps_prev[i], nmaps_prev[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::getIncrementalTransformation(Sophus::SE3d & T_prev_curr, int threads, int blocks)
{
    iterations[0] = 10;
    iterations[1] = 5;
    iterations[2] = 4;

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        for(int j = 0; j < iterations[i]; j++)
        {
            float residual_inliers[2];
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            estimateStep(T_prev_curr.rotationMatrix().cast<float>().eval(),
                         T_prev_curr.translation().cast<float>().eval(),
                         vmaps_curr[i],
                         nmaps_curr[i],
                         intr(i),
                         vmaps_prev[i],
                         nmaps_prev[i],
                         dist_thresh,
                         angle_thresh,
                         sumData,
                         outData,
                         A_icp.data(),
                         b_icp.data(),
                         &residual_inliers[0],
                         threads,
                         blocks);

            lastError = sqrt(residual_inliers[0]) / residual_inliers[1];
            lastInliers = residual_inliers[1];

            const Eigen::Matrix<double, 6, 1> update = A_icp.cast<double>().ldlt().solve(b_icp.cast<double>());

            T_prev_curr = Sophus::SE3d::exp(update) * T_prev_curr;
        }
    }
}
