#pragma once

#include <cuda_runtime.h>
#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

typedef unsigned char uchar;

namespace df {


template <typename ... DebugArgTs>
struct PixelDebugger {

    inline __device__ static void debugPixel(const Eigen::Vector2i & pixel,
                                             const Eigen::UnalignedVec4<uchar> & color,
                                             DebugArgTs ... /*debugArgs*/) { }

};

template <>
struct PixelDebugger<DeviceTensor2<Eigen::UnalignedVec4<unsigned char> > > {

    inline __device__ static void debugPixel(const Eigen::Vector2i & pixel,
                                             const Eigen::UnalignedVec4<uchar> & color,
                                             DeviceTensor2<Eigen::UnalignedVec4<unsigned char> > debugArg) {

        debugArg(pixel) = color;

    }

};



} // namespace df
