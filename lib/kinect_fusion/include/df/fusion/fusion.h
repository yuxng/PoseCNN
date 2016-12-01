#pragma once

#include <df/util/tensor.h>

namespace df {

template <typename Scalar,
          typename VoxelGridT,
          typename TransformerT,
          typename CameraModelT,
          typename DepthT>
void fuseFrame(VoxelGridT & voxelGrid,
               TransformerT & transformer,
               CameraModelT & cameraModel,
               Tensor<2,DepthT,DeviceResident> & depthMap,
               const Scalar truncationDistance);

} // namespace df
