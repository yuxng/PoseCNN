#include <df/camera/rig.h>

#include <df/camera/cameraFactory.h>

namespace df {

template <typename T>
inline Sophus::SE3<T> poseFromJson(const picojson::value & poseSpec) {

    std::cout << "pose: " << poseSpec.serialize(true) << std::endl;

    if (poseSpec.size() != 3) {
        throw std::runtime_error("pose spec must have 3 rows");
    }

    Eigen::Matrix<T,4,4> M = Eigen::Matrix<T,4,4>::Identity();
    for (int r = 0; r < 3; ++r) {
        const picojson::value & rowSpec = poseSpec[r];
        if (rowSpec.size() != 4) {
            throw std::runtime_error("each row in pose spec must have 4 values");
        }
        for (int c = 0; c < 4; ++c) {
            M(r,c) = atof(rowSpec[c].to_str().c_str());
        }
    }

    return Sophus::SE3<T>(M);

}

template <typename T>
Rig<T>::Rig(const picojson::value & rigSpec) {

    const picojson::value & camsSpec = rigSpec["camera"];

    const std::size_t nCameras = camsSpec.size();

    cameras_.resize(nCameras);
    transformsCameraToRig_.resize(nCameras);

    CameraFactory<T> & cameraFactory = CameraFactory<T>::instance();

    for (std::size_t i = 0; i < nCameras; ++i) {

        const picojson::value & camSpec = camsSpec[i];
        if (!camSpec.contains("camera_model")) {
            throw std::runtime_error("camera spec does not contain a camera model");
        }

        cameras_[i].reset(cameraFactory.createCamera(camSpec["camera_model"]));

        if (!camSpec.contains("pose")) {
            throw std::runtime_error("camera spec does not contain a pose");
        }

        transformsCameraToRig_[i] = poseFromJson<T>(camSpec["pose"]);

    }
}

template class Rig<float>;
template class Rig<double>;

} // namespace df
