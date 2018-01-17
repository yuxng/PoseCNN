#pragma once

#include <sophus/se3.hpp>

#include <cuda_runtime.h>

namespace df {

template <typename Scalar>
class RigidTransformer {
public:

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Sophus::SE3<Scalar> Transform;

    inline __host__ __device__ Vec3 transformWorldToLive(Vec3 world) const {
        return transform_*world;
    }

    inline __host__ __device__ Vec3 transformLiveToWorld(Vec3 live) const {
        return transform_.inverse()*live;
    }

    inline __host__ __device__ Transform worldToLiveTransformation() const {
        return transform_;
    }

    inline __host__ __device__ Transform liveToWorldTransformation() const {
        return transform_.inverse();
    }

    inline __host__ __device__ void setWorldToLiveTransformation(const Transform & transform) {
        transform_ = transform;
    }

    inline __host__ __device__ void setLiveToWorldTransformation(const Transform & transform) {
        transform_ = transform.inverse();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef RigidTransformer<Scalar> DeviceModule;

    inline DeviceModule deviceModule() const {

        return *this;

    }

private:

    Transform transform_;

};

} // namespace df
