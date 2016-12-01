#pragma once

#include <df/util/tensor.h>
#include <df/voxel/voxelGrid.h>

namespace df {

template <typename Scalar>
void computeTriangularFaceNormals(const Tensor<2,Scalar,DeviceResident> & vertices,
                                  Tensor<2,Scalar,DeviceResident> & normals);

template <typename Scalar,
          typename VoxelT>
void computeSignedDistanceGradientNormals(const Tensor<2,Scalar,DeviceResident> & vertices,
                                          Tensor<2,Scalar,DeviceResident> & normals,
                                          VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid);

} // namespace df
