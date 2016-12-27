#pragma once

#include <Eigen/Core>

#include <df/voxel/movingAverage.h>

namespace df {

struct TsdfVoxel : public MovingAverageVoxel<float,float> {

    inline __host__ __device__ float signedDistanceValue() const {
        return value();
    }

    inline __host__ __device__
    static TsdfVoxel zero() {
        TsdfVoxel voxel;
        voxel.weight_ = 0.f;
        voxel.weightedAverage_ = 0.f;
//        voxel.sdfAndWeight = Eigen::Vector2f(0,0);
        return voxel;
    }

//    template <typename T>
//    inline __host__ __device__ void fuse(const T signedDistance, const T weight, const T maxWeight) {
//        const float currentWeight = sdfAndWeight(1);
//        const float newWeight = currentWeight + weight;
//        sdfAndWeight(0) = (currentWeight / newWeight ) * sdfAndWeight(0) + (weight / newWeight) * signedDistance;
//        sdfAndWeight(1) = newWeight < maxWeight ? newWeight : maxWeight;
//    }

//    inline __host__ __device__ float signedDistanceValue() const {
//        return sdfAndWeight(0);
//    }

//    inline __host__ __device__ float weight() const {
//        return sdfAndWeight(1);
//    }

//    Eigen::Vector2f sdfAndWeight;

};

} // namespace df
