#pragma once

#include <df/util/tensor.h>

namespace df {

template <typename Scalar, typename ScalarComp>
void radiallySymmetricBilateralFilter(DeviceTensor2<Scalar> & destination, const DeviceTensor2<Scalar> & source,
                                      DeviceTensor1<ScalarComp> & halfKernel, const ScalarComp sigmaInput);

template <typename Scalar, typename ScalarComp>
void radiallySymmetricBilateralFilterAndDownsampleBy2(DeviceTensor2<Scalar> & destination, const DeviceTensor2<Scalar> & source,
                                                      DeviceTensor1<ScalarComp> & halfKernel, const ScalarComp sigmaInput);


} // namespace df
