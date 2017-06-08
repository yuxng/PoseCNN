#pragma once

#include <pangolin/utils/picojson.h>
#include <df/camera/cameraModel.h>
#include <Eigen/Core>

namespace df {

template <typename T>
class CameraBase {
public:

    CameraBase(const picojson::value & cameraSpec);

    inline unsigned int width() const { return width_; }

    inline unsigned int height() const { return height_; }

    virtual const T * params() const = 0;

    virtual Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1> point3d) const = 0;

    virtual Eigen::Matrix<T,3,1> unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const = 0;

private:

    int width_;
    int height_;

};

template <template <typename> class ModelT, typename T>
class Camera : public CameraBase<T> {
public:

    Camera(const picojson::value & cameraSpec)
        : CameraBase<T>(cameraSpec), model_(cameraSpec) { }

    inline const T * params() const {
        return model_.params();
    }

    inline Eigen::Matrix<T,2,1> project(const Eigen::Matrix<T,3,1> point3d) const {
        return model_.project(point3d);
    }

    inline Eigen::Matrix<T,3,1> unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const {
        return model_.unproject(point2d,depth);
    }

    inline ModelT<T> & model() { return model_; }

    inline const ModelT<T> & model() const { return model_; }

private:
    ModelT<T> model_;
};


} // namespace df
