#pragma once

#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

namespace df {

template <typename Scalar,
          template <typename> class CamModelT>
void backproject(const HostTensor2<Scalar> & depthMap,
                 HostTensor2<Eigen::UnalignedVec3<Scalar> > & vertMap,
                 CamModelT<Scalar> & cameraModel);

template <typename Scalar,
          template <typename> class CamModelT>
void backproject(const DeviceTensor2<Scalar> & depthMap,
                 DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & vertMap,
                 CamModelT<Scalar> & cameraModel);


} // namespace df
