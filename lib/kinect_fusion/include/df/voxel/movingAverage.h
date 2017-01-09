#pragma once

#include <cuda_runtime.h>

namespace df {

template <typename Scalar, typename AveragedType>
struct MovingAverageVoxel {

    typedef AveragedType ObservationType;

    inline __host__ __device__ void fuse(const AveragedType thisValue, const Scalar thisWeight, const Scalar maxWeight) {
        const Scalar currentWeight = weight();
        const Scalar newWeight = currentWeight + thisWeight;
        weightedAverage_ = ( currentWeight / newWeight ) * weightedAverage_ +
                ( thisWeight / newWeight ) * thisValue;
        weight_ = newWeight > maxWeight ? maxWeight : newWeight; //min(newWeight,maxWeight);
    }

    inline __host__ __device__ AveragedType value() const {
        return weightedAverage_;
    }

    inline __host__ __device__ Scalar weight() const {
        return weight_;
    }

    Scalar weight_;
    AveragedType weightedAverage_;

};


} // namespace df
