#pragma once

#include <Eigen/Core>

#include <df/voxel/movingAverage.h>

namespace df {

struct ProbabilityVoxel : public MovingAverageVoxel<float,Eigen::Matrix<float, 10, 1, Eigen::DontAlign> > {

    inline __host__ __device__
    static ProbabilityVoxel zero() {
        ProbabilityVoxel voxel;
        voxel.weight_ = 0.f;
        voxel.weightedAverage_ = Eigen::Matrix<float,10,1,Eigen::DontAlign>::Zero();
        return voxel;
    }

    inline __host__ __device__ Eigen::Matrix<float,10,1,Eigen::DontAlign> probabilityValue() const {
        return value();
    }

};

} // namespace df
