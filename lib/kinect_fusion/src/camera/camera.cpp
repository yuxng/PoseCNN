#include <df/camera/camera.h>

namespace df {

template <typename T>
CameraBase<T>::CameraBase(const pangolin::json::value & cameraSpec) {

    if (!cameraSpec.contains("width")) {
        throw std::runtime_error("camera spec does not give the width of the image");
    }

    if (!cameraSpec.contains("height")) {
        throw std::runtime_error("camera spec does not give the height of the camera");
    }

    const pangolin::json::value & widthSpec = cameraSpec["width"];
    const pangolin::json::value & heightSpec = cameraSpec["height"];

    width_ = atoi(widthSpec.to_str().c_str());
    height_ = atoi(heightSpec.to_str().c_str());

}

template class CameraBase<float>;
template class CameraBase<double>;

} // namespace df
