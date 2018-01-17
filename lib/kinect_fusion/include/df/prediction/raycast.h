#pragma once

#include <df/util/tensor.h>
#include <df/voxel/voxelGrid.h>

#include <sophus/se3.hpp>

namespace df {

template <typename Scalar,
          typename VoxelT,
          typename CameraModelT>
void raycast(Tensor<3,Scalar,DeviceResident> & predictedVertices,
             Tensor<3,Scalar,DeviceResident> & predictedNormals,
             const VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid,
             const CameraModelT & cameraModel,
             const Sophus::SE3<Scalar> & transformWorldToPrediction);


} // namespace df
