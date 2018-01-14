#pragma once

#include <memory>
#include <vector>
#include <pangolin/utils/picojson.h>
#include <df/camera/camera.h>
#include <sophus/se3.hpp>

namespace df {

template <typename T>
class Rig {
public:

    Rig(const picojson::value & rigSpec);

    const std::size_t numCameras() const {
        return cameras_.size();
    }

    inline const CameraBase<T> & camera(const std::size_t index) const {
        return *cameras_[index];
    }

    inline const Sophus::SE3<T> & transformCameraToRig(const std::size_t index) const {
        return transformsCameraToRig_[index];
    }

private:

    std::vector<std::shared_ptr<CameraBase<T> > > cameras_;
    std::vector<Sophus::SE3<T> > transformsCameraToRig_;

};


} // namespace df
