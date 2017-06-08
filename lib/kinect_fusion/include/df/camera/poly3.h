#pragma once

#include <df/camera/cameraModel.h>
#include <df/camera/cameraFactory.h>

namespace df {

template <typename Scalar>
class Poly3CameraModel;

namespace internal {

template <>
struct CameraModelTraits<Poly3CameraModel> {

    static constexpr int NumParams = 7;

};

} // namespace internal

template <typename Scalar>
class Poly3CameraModel : public CameraModel<Poly3CameraModel, Scalar> {
public:

    Poly3CameraModel(const picojson::value & cameraSpec)
        : CameraModel<Poly3CameraModel,Scalar>(cameraSpec) { }

    template <typename T2>
    Poly3CameraModel(const CameraModel<Poly3CameraModel,T2> & other)
        : CameraModel<Poly3CameraModel,Scalar>(other) { }

    Poly3CameraModel(const Eigen::Matrix<Scalar,7,1,Eigen::DontAlign> & params)
        : CameraModel<Poly3CameraModel,Scalar>(params) { }

    inline __host__ __device__ Scalar focalLengthX() const {
        return this->params()[0];
    }

    inline __host__ __device__ Scalar focalLengthY() const {
        return this->params()[1];
    }

    inline __host__ __device__ Scalar principalPointX() const {
        return this->params()[2];
    }

    inline __host__ __device__ Scalar principalPointY() const {
        return this->params()[3];
    }

    inline __host__ __device__ Scalar k1() const {
        return this->params()[4];
    }

    inline __host__ __device__ Scalar k2() const {
        return this->params()[5];
    }

    inline __host__ __device__ Scalar k3() const {
        return this->params()[6];
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> focalLength() const {
        return Eigen::Matrix<Scalar,2,1>(focalLengthX(),focalLengthY());
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> principalPoint() const {
        return Eigen::Matrix<Scalar,2,1>(principalPointX(),principalPointY());
    }

    inline Eigen::Matrix<Scalar,2,1> __host__ __device__ project(const Eigen::Matrix<Scalar,3,1> point3d) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->dehomogenize(point3d);

        const Scalar radius2 = dehomog.squaredNorm();

        const Scalar radius4 = radius2 * radius2;

        const Scalar radius6 = radius4 * radius2;

        const Scalar distortionFactor = Scalar(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

        const Eigen::Matrix<Scalar,2,1> distorted = distortionFactor * dehomog;

        return this->applyFocalLengthAndPrincipalPoint(distorted,focalLength(),principalPoint());

    }

    inline Eigen::Matrix<Scalar,3,1> __host__ __device__ unproject(const Eigen::Matrix<Scalar,2,1> point2d, const Scalar depth) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->unapplyFocalLengthAndPrincipalPoint(point2d,focalLength(),principalPoint());

        const Scalar radiusInit = dehomog.norm();

        if (radiusInit > Scalar(0)) {

            Scalar radius = radiusInit;
            for (int i = 0; i < maxUnprojectionIters; ++i) {

                const Scalar radius2 = radius*radius;

                const Scalar radius4 = radius2*radius2;

                const Scalar radius6 = radius4*radius2;

                const Scalar distortionFactor = Scalar(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

                const Scalar distortionFactor2 = 2*radius2*(k1() + 2*k2()*radius2 + 3*k3()*radius4);

                const Scalar distortionFactor3 = distortionFactor + distortionFactor2;

                const Scalar derivative = (radius * distortionFactor - radiusInit) * 2 * distortionFactor3;

                const Scalar derivative2 = (4 * radius * ( radius * distortionFactor - radiusInit) *
                                            (3 * k1() + 10*k2()*radius2 + 21*k3()*radius4) +
                                            2*distortionFactor3*distortionFactor3);

                const Scalar delta = derivative / derivative2;

                radius -= delta;
            }

            const Scalar undistortionFactor = radius / radiusInit;

            const Eigen::Matrix<Scalar,2,1> undistorted = dehomog*undistortionFactor;

            const Eigen::Matrix<Scalar,2,1> scaled = undistorted * depth;

            return Eigen::Matrix<Scalar,3,1>(scaled(0),scaled(1),depth);

        } else {

            return Eigen::Matrix<Scalar,3,1>(dehomog(0)*depth,dehomog(1)*depth,depth);

        }

    }

    inline Poly3CameraModel<Scalar> downsampleBy2() {

        Eigen::Matrix<Scalar,7,1,Eigen::DontAlign> downsampledParams = this->params_;

        downsampledParams.template head<2>() /= Scalar(2);

        downsampledParams.template segment<2>(2) /= Scalar(2);
        downsampledParams.template segment<2>(2) -= Eigen::Matrix<Scalar,2,1>(1 / Scalar(4), 1 / Scalar(4));

        return Poly3CameraModel<Scalar>(downsampledParams);

    }

private:

    static constexpr int maxUnprojectionIters = 5;

};

} // namespace df
