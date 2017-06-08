#pragma once

#include <df/camera/cameraModel.h>
#include <df/camera/cameraFactory.h>

namespace df {

template <typename T>
class LinearCameraModel;

namespace internal {

template <>
struct CameraModelTraits<LinearCameraModel> {

    static constexpr int NumParams = 4;

};

} // namespace internal

template <typename T>
class LinearCameraModel : public CameraModel<LinearCameraModel, T> {
public:

    LinearCameraModel(const picojson::value & cameraSpec)
        : CameraModel<LinearCameraModel,T>(cameraSpec) { }

    template <typename T2>
    LinearCameraModel(const CameraModel<LinearCameraModel,T2> & other)
        : CameraModel<LinearCameraModel,T>(other) { }

    inline __host__ __device__ T focalLengthX() const {
        return this->params()[0];
    }

    inline __host__ __device__ T focalLengthY() const {
        return this->params()[1];
    }

    inline __host__ __device__ T principalPointX() const {
        return this->params()[2];
    }

    inline __host__ __device__ T principalPointY() const {
        return this->params()[3];
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> focalLength() const {
        return Eigen::Matrix<T,2,1>(focalLengthX(),focalLengthY());
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> principalPoint() const {
        return Eigen::Matrix<T,2,1>(principalPointX(),principalPointY());
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1> point3d) const {

        const Eigen::Matrix<T,2,1> dehomog = this->dehomogenize(point3d);
        return this->applyFocalLengthAndPrincipalPoint(dehomog,focalLength(),principalPoint());

    }

    inline __host__ __device__ Eigen::Matrix<T,3,1> unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const {

        const Eigen::Matrix<T,2,1> dehomog = this->unapplyFocalLengthAndPrincipalPoint(point2d,focalLength(),principalPoint());
        const Eigen::Matrix<T,2,1> scaled = dehomog * depth;

        return Eigen::Matrix<T,3,1>(scaled(0),scaled(1),depth);

    }

protected:

};

} // namespace df
