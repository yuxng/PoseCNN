#include <df/voxel/voxelGrid.h>

#include <df/voxel/tsdf.h>
#include <df/util/cudaHelpers.h>

#include <iostream>

namespace df {

namespace internal {

template <typename VoxelT>
__global__ void fillVoxelGridKernel(Tensor<3,VoxelT,DeviceResident> grid, const VoxelT value) {

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = threadIdx.y + blockDim.y * blockIdx.y;
    const uint z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < grid.dimensionSize(0)) && (y < grid.dimensionSize(1)) && (z < grid.dimensionSize(2))) {

        grid(x,y,z) = value;

    }

}

template <>
template <typename VoxelT>
void VoxelGridFiller<DeviceResident>::fill(Tensor<3,VoxelT,DeviceResident> & grid, const VoxelT & value) {

    dim3 block(16,16,4);
    dim3 threadGrid(intDivideAndCeil(grid.dimensionSize(0),block.x),
                    intDivideAndCeil(grid.dimensionSize(1),block.y),
                    intDivideAndCeil(grid.dimensionSize(2),block.z));

    // TODO
    fillVoxelGridKernel<<<threadGrid,block>>>(grid,value);

}

template void VoxelGridFiller<DeviceResident>::fill(Tensor<3,TsdfVoxel,DeviceResident> &, const TsdfVoxel &);

template void VoxelGridFiller<DeviceResident>::fill(Tensor<3,Eigen::Matrix<int,4,1,Eigen::DontAlign>,DeviceResident> &, const Eigen::Matrix<int,4,1,Eigen::DontAlign> &);

} // namespace internal

} // namespace df
