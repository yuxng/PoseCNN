#pragma once

#include <df/voxel/voxelGrid.h>

namespace df {

template <typename Scalar,
          typename VoxelT>
void extractSurface(ManagedTensor<2,Scalar,DeviceResident> & vertices,
                    const VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid,
                    const Scalar weightThreshold);

template <typename Scalar>
uint weldVertices(const Tensor<2,Scalar,DeviceResident> & vertices,
                  Tensor<2,Scalar,DeviceResident> & weldedVertices,
                  ManagedTensor<1,int,DeviceResident> & indices);

void initMarchingCubesTables();

} // namespace df
