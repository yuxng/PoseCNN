#pragma once

#include <df/camera/cameraModel.h>
#include <df/camera/cameraFactory.h>

namespace df {

template <typename T>
class Poly3CameraModel;

namespace internal {

template <>
struct CameraModelTraits<Poly3CameraModel> {

    static constexpr int NumParams = 7;

};

} // namespace internal

template <typename T>
class Poly3CameraModel : public CameraModel<Poly3CameraModel, T> {
public:

    Poly3CameraModel(const pangolin::json::value & cameraSpec)
        : CameraModel<Poly3CameraModel,T>(cameraSpec) { }

    template <typename T2>
    Poly3CameraModel(const CameraModel<Poly3CameraModel,T2> & other)
        : CameraModel<Poly3CameraModel,T>(other) { }

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

    inline __host__ __device__ T k1() const {
        return this->params()[4];
    }

    inline __host__ __device__ T k2() const {
        return this->params()[5];
    }

    inline __host__ __device__ T k3() const {
        return this->params()[6];
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> focalLength() const {
        return Eigen::Matrix<T,2,1>(focalLengthX(),focalLengthY());
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> principalPoint() const {
        return Eigen::Matrix<T,2,1>(principalPointX(),principalPointY());
    }

    inline Eigen::Matrix<T,2,1> __host__ __device__ project(const Eigen::Matrix<T,3,1> point3d) const {

        const Eigen::Matrix<T,2,1> dehomog = this->dehomogenize(point3d);

        const T radius2 = dehomog.squaredNorm();

        const T radius4 = radius2 * radius2;

        const T radius6 = radius4 * radius2;

        const T distortionFactor = T(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

        const Eigen::Matrix<T,2,1> distorted = distortionFactor * dehomog;

        return this->applyFocalLengthAndPrincipalPoint(distorted,focalLength(),principalPoint());

    }

    inline Eigen::Matrix<T,3,1> __host__ __device__ unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const {

        const Eigen::Matrix<T,2,1> dehomog = this->unapplyFocalLengthAndPrincipalPoint(point2d,focalLength(),principalPoint());

        const T radiusInit = dehomog.norm();

        T radius = radiusInit;
        for (int i = 0; i < maxUnprojectionIters; ++i) {

            const T radius2 = radius*radius;

            const T radius4 = radius2*radius2;

            const T radius6 = radius4*radius2;

            const T distortionFactor = T(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

            const T distortionFactor2 = 2*radius2*(k1() + 2*k2()*radius2 + 3*k3()*radius4);

            const T distortionFactor3 = distortionFactor + distortionFactor2;

            const T derivative = (radius * distortionFactor - radiusInit) * 2 * distortionFactor3;

            const T derivative2 = (4 * radius * ( radius * distortionFactor - radiusInit) *
                                   (3 * k1() + 10*k2()*radius2 + 21*k3()*radius4) +
                                   2*distortionFactor3*distortionFactor3);

            const T delta = derivative / derivative2;

            radius -= delta;
        }

        const T undistortionFactor = radius / radiusInit;

        const Eigen::Matrix<T,2,1> undistorted = dehomog*undistortionFactor;

        const Eigen::Matrix<T,2,1> scaled = undistorted * depth;

        return Eigen::Matrix<T,3,1>(scaled(0),scaled(1),depth);

    }

private:

    static constexpr int maxUnprojectionIters = 5;

};

} // namespace df
