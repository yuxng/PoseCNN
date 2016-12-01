#pragma once

#include <cuda_runtime.h>

#include <Eigen/Core>

namespace df {

struct TsdfVoxel {

    inline __host__ __device__
    static TsdfVoxel zero() {
        TsdfVoxel voxel;
        voxel.sdfAndWeight = Eigen::Vector2f(0,0);
        return voxel;
    }

    template <typename T>
    inline __host__ __device__ void fuse(const T signedDistance, const T weight, const T maxWeight) {
        const float currentWeight = sdfAndWeight(1);
        const float newWeight = currentWeight + weight;
        sdfAndWeight(0) = (currentWeight / newWeight ) * sdfAndWeight(0) + (weight / newWeight) * signedDistance;
        sdfAndWeight(1) = newWeight < maxWeight ? newWeight : maxWeight;
    }

    inline __host__ __device__ float signedDistanceValue() const {
        return sdfAndWeight(0);
    }

    inline __host__ __device__ float weight() const {
        return sdfAndWeight(1);
    }

    Eigen::Vector2f sdfAndWeight;

};

inline __host__ __device__
TsdfVoxel operator+(const TsdfVoxel & a, const TsdfVoxel & b) {
    return { a.sdfAndWeight + b.sdfAndWeight };
}

template <typename Scalar>
inline __host__ __device__
TsdfVoxel operator*(const Scalar a, const TsdfVoxel & b) {
    return { a*b.sdfAndWeight };
}

template <typename Scalar>
inline __host__ __device__
TsdfVoxel operator*(const TsdfVoxel & a, const Scalar b) {
    return b*a;
}

} // namespace df
