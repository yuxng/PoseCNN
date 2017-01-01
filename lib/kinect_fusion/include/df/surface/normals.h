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

template <typename Scalar, int D>
void computeVertMapNormals(const DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > & vertMap,
                           DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > & normMap);

} // namespace df
