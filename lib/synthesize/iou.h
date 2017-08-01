#pragma once

#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

namespace df {

float iou(const DeviceTensor2<int> & labelMap, DeviceTensor2<int> & interMap, DeviceTensor2<int> & unionMap,
                DeviceTensor2<Eigen::UnalignedVec4<float> > & vertMap, int classID);


} // namespace df
