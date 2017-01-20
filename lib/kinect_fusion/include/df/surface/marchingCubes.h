#pragma once

#include <df/voxel/voxelGrid.h>
#include <df/voxel/tsdf.h>
#include <df/voxel/probability.h>
#include <df/voxel/compositeVoxel.h>

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

template <typename TransformerT,
          typename DepthCameraModelT,
          typename DepthT>
void computeLabels(const TransformerT & transformer,
                   const DepthCameraModelT & depthCameraModel,
                   const DeviceTensor2<DepthT> & depthMap,
                   const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident> & voxelGrid,
                   DeviceTensor2<int> & labels, 
                   DeviceTensor2<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & label_colors, 
                   const DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & class_colors);

} // namespace df
