#include <df/image/backprojection.h>

#include <assert.h>
#include <Eigen/Core>

#include <df/camera/cameraModel.h>

#include <df/util/macros.h>

namespace df {

template <typename Scalar,
          template <typename> class CamModelT>
void backproject(const HostTensor2<Scalar> & depthMap,
                 HostTensor2<Eigen::UnalignedVec3<Scalar> > & vertMap,
                 CamModelT<Scalar> & cameraModel) {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;

    const uint width = depthMap.dimensionSize(0);
    const uint height = depthMap.dimensionSize(1);

    for (uint y = 0; y < height; ++y) {
        for (uint x = 0; x < width; ++x) {
            const Scalar depth = depthMap(x,y);
            vertMap(x,y) = cameraModel.unproject(Vec2(x,y),depth);
        }
    }

}

#define BACKPROJECT_EXPLICIT_INSTANTIATION(type,camera)                   \
    template void backproject(const HostTensor2<type> &,                  \
                              HostTensor2<Eigen::UnalignedVec3<type> > &, \
                              camera##CameraModel<type> &)

ALL_CAMERAS_AND_TYPES_INSTANTIATION(BACKPROJECT_EXPLICIT_INSTANTIATION);

} // namespace df
