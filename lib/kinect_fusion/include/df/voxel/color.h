#pragma once

#include <Eigen/Core>

#include <df/voxel/movingAverage.h>

namespace df {

struct ColorVoxel : public MovingAverageVoxel<float,Eigen::Matrix<float,3,1,Eigen::DontAlign> > {

    inline __host__ __device__
    static ColorVoxel zero() {
        ColorVoxel voxel;
        voxel.weight_ = 0.f;
        voxel.weightedAverage_ = Eigen::Matrix<float,3,1,Eigen::DontAlign>::Zero();
        return voxel;
    }

    inline __host__ __device__ Eigen::Matrix<float,3,1,Eigen::DontAlign> colorValue() const {
        return value();
    }

};

} // namespace df
