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

template <typename Scalar, typename VoxelT>
void computeColors(const Tensor<2, Scalar, DeviceResident> & vertices, int* labels,
                   unsigned char* class_colors, const VoxelGrid<Scalar, VoxelT, DeviceResident> & voxelGrid,
                   Tensor<2, unsigned char, DeviceResident> & colors, int dimension, int num_classes);

} // namespace df
