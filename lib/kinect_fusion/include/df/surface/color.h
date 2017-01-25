#pragma once

#include <df/util/tensor.h>
#include <df/voxel/color.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/tsdf.h>
#include <df/voxel/voxelGrid.h>

namespace df {

template <typename Scalar, typename VoxelT>
void computeSurfaceColors(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
                          DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & colors,
                          const DeviceVoxelGrid<Scalar,VoxelT> & voxelGrid,
                          const DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > &);

} // namespace df
