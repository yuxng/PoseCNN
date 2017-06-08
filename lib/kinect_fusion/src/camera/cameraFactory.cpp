#include <df/camera/cameraFactory.h>

namespace df {

template <typename T>
CameraFactory<T> & CameraFactory<T>::instance() {
    static CameraFactory<T> obj;
    return obj;
}

template <typename T>
CameraBase<T> * CameraFactory<T>::createCamera(const picojson::value & cameraSpec) {

    std::cout << cameraSpec << std::endl;
    std::cout << cameraSpec["type"] << std::endl;

    const std::string type = cameraSpec["type"].get<std::string>();

    if (cameraCreators_.find(type) == cameraCreators_.end()) {
        throw std::runtime_error("unknown camera model: " + type);
    }

    return cameraCreators_[type](cameraSpec);
}

template <typename T>
void CameraFactory<T>::registerCameraCreator(const std::string name, CameraCreator creator) {

    cameraCreators_[name] = creator;

}

template class CameraFactory<float>;
template class CameraFactory<double>;

} // namespace df
