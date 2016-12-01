#pragma once

#include <pangolin/utils/picojson.h>

#include <Eigen/Core>

#include <cuda_runtime.h>

namespace df {

namespace internal {

template <template <typename> class ModelT>
struct CameraModelTraits;

} // namespace internal

template <template <typename> class ModelT, typename T>
class CameraModel {
public:

    CameraModel(const pangolin::json::value & cameraSpec) {

        if (cameraSpec["params"].size() != numParams()) {
            throw std::runtime_error("wrong number of parameters for specifiec model (" +
                                     std::to_string(cameraSpec.size()) + " vs " + std::to_string(numParams()));
        }

        std::cout << "params: ";
        for (unsigned int i = 0; i < numParams(); ++i) {
            std::cout << cameraSpec["params"][i] << "  ";
            params_[i] = atof(cameraSpec["params"][i].to_str().c_str());
        } std::cout << std::endl;

    }

    template <typename T2>
    CameraModel(const CameraModel<ModelT,T2> & other) {
        for (unsigned int i = 0; i < numParams(); ++i) {
            params_[i] = other.params()[i];
        }
    }

    template <typename T2>
    inline ModelT<T2> cast() const {
        return ModelT<T2>(*this);
    }

    inline operator ModelT<T> () { return *this; }

    inline operator const ModelT<T> () const { return *this; }

    inline static unsigned int numParams() { return internal::CameraModelTraits<ModelT>::NumParams; }

    inline __host__ __device__ const T * params() const {
        return params_;
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1> point3d) const {
        return static_cast<const ModelT<T> *>(this)->project(point3d);
    }

    inline __host__ __device__ Eigen::Matrix<T,3,1> unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const {
        return static_cast<const ModelT<T> *>(this)->unproject(point2d,depth);
    }

protected:

    inline __host__ __device__ Eigen::Matrix<T,2,1> dehomogenize(const Eigen::Matrix<T,3,1> point3d) const {
        return Eigen::Matrix<T,2,1>(point3d(0)/point3d(2),
                                    point3d(1)/point3d(2));
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> applyFocalLengthAndPrincipalPoint(const Eigen::Matrix<T,2,1> dehomogPoint,
                                                                  const Eigen::Matrix<T,2,1> focalLength,
                                                                  const Eigen::Matrix<T,2,1> principalPoint) const {
        return Eigen::Matrix<T,2,1>(dehomogPoint(0)*focalLength(0),
                                    dehomogPoint(1)*focalLength(1)) + principalPoint;
    }

    inline __host__ __device__ Eigen::Matrix<T,2,1> unapplyFocalLengthAndPrincipalPoint(const Eigen::Matrix<T,2,1> point2d,
                                                                    const Eigen::Matrix<T,2,1> focalLength,
                                                                    const Eigen::Matrix<T,2,1> principalPoint) const {
        const Eigen::Matrix<T,2,1> centered = point2d - principalPoint;
        return Eigen::Matrix<T,2,1>(centered(0)/focalLength(0),
                                    centered(1)/focalLength(1));
    }

private:

    T params_[internal::CameraModelTraits<ModelT>::NumParams];

};

} // namespace df
