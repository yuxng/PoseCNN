#pragma once

#include <functional>
#include <map>

#include <pangolin/utils/picojson.h>

#include <df/camera/camera.h>

namespace df {

template <typename T>
class CameraFactory {
public:

    static CameraFactory & instance();

    typedef std::function<CameraBase<T> *(const picojson::value &)> CameraCreator;

    CameraBase<T> * createCamera(const picojson::value & cameraSpec);

    void registerCameraCreator(const std::string name, CameraCreator creator);

private:

    CameraFactory() { }
    CameraFactory(const CameraFactory &);
    CameraFactory & operator=(const CameraFactory &);
    ~CameraFactory() { }

    std::map<std::string,CameraCreator> cameraCreators_;

};

namespace internal {

template <typename T>
struct CameraModelRegistration {
CameraModelRegistration(const std::string name,
                        typename CameraFactory<T>::CameraCreator creator) {
    CameraFactory<T>::instance().registerCameraCreator(name,creator);
    std::cout << "registered " << name << std::endl;
}
};

#define REGISTER_CAMERA_MODEL(name)                                                                                        \
    template <typename T>                                                                                                  \
    CameraBase<T> * create##name##CameraModel(const picojson::value & cameraSpec) {                                  \
        return new Camera<name##CameraModel,T>(cameraSpec);                                                                \
    }                                                                                                                      \
    static internal::CameraModelRegistration<float> name##CameraRegistration_f(#name, create##name##CameraModel<float>);   \
    static internal::CameraModelRegistration<double> name##CameraRegistration_d(#name, create##name##CameraModel<double>)

} // namespace internal

} // namespace df
