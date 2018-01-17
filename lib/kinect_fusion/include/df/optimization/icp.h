#pragma once

#include <sophus/se3.hpp>

#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

#include <df/optimization/linearSystems.h>
#include <thrust/device_vector.h>

typedef unsigned char uchar;

namespace df {

template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
Sophus::SE3<Scalar> icp(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                             const CameraModelT & cameraModel,
                             const Sophus::SE3<Scalar> & predictionPose,
                             const Eigen::Matrix<Scalar,2,1> & depthRange,
                             const Scalar maxError,
                             const uint numIterations,
                             DebugArgsT ... debugArgs);

template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
Sophus::SE3<Scalar> icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                             const CameraModelT & cameraModel,
                             const Sophus::SE3<Scalar> & predictionPose,
                             const Eigen::Matrix<Scalar,2,1> & depthRange,
                             const Scalar maxError,
                             const uint numIterations,
                             DebugArgsT ... debugArgs);

namespace internal {

template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
LinearSystem<Scalar,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                                    const CameraModelT & cameraModel,
                                    const Sophus::SE3<Scalar> & predictionPose,
                                    const Eigen::Matrix<Scalar,6,1>& initialPose,
                                    const Eigen::Matrix<Scalar,2,1> & depthRange,
                                    const Scalar maxError,
                                    const dim3 grid,
                                    const dim3 block,
                                    DebugArgsT ... debugArgs);

} // namespace internal

} // namespace df
