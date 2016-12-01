#pragma once

#include <sophus/se3.hpp>

#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

#include <df/optimization/linearSystems.h>
#include <thrust/device_vector.h>

namespace df {

template <typename Scalar,
          typename CameraModelT,
          int DPred>
Sophus::SE3Group<Scalar> icp(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                             const CameraModelT & cameraModel,
                             const Sophus::SE3Group<Scalar> & predictionPose,
                             const Eigen::Matrix<Scalar,2,1> & depthRange,
                             const Scalar maxError,
                             const uint numIterations);

namespace internal {

template <typename Scalar,
          typename CameraModelT,
          int DPred>
LinearSystem<Scalar,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                                    const CameraModelT & cameraModel,
                                    const Sophus::SE3Group<Scalar> & predictionPose,
                                    const Eigen::Matrix<Scalar,2,1> & depthRange,
                                    const Scalar maxError,
                                    const dim3 grid,
                                    const dim3 block);

} // namespace internal

} // namespace df
