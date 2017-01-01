#include <df/image/backprojection.h>

#include <df/util/cudaHelpers.h>
#include <df/util/macros.h>

namespace df {

template <typename Scalar,
          typename CamModelT>
__global__ void backprojectKernel(const DeviceTensor2<Scalar> depthMap,
                                  DeviceTensor2<Eigen::UnalignedVec3<Scalar> > vertMap,
                                  CamModelT cameraModel) {

    typedef Eigen::Matrix<Scalar,3,1> Vec3;
    typedef Eigen::Matrix<Scalar,2,1> Vec2;

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if ( (x < depthMap.dimensionSize(0)) && (y < depthMap.dimensionSize(1))) {

        const Scalar depth = depthMap(x,y);
        vertMap(x,y) = cameraModel.unproject(Vec2(x,y),depth);

    }

}


template <typename Scalar,
          template <typename> class CamModelT>
void backproject(const DeviceTensor2<Scalar> & depthMap,
                 DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & vertMap,
                 CamModelT<Scalar> & cameraModel) {

    const dim3 block(16,16,1);
    const dim3 grid(intDivideAndCeil(depthMap.dimensionSize(0),block.x),
                    intDivideAndCeil(depthMap.dimensionSize(1),block.y),
                    1);
    backprojectKernel<<<grid,block>>>(depthMap,vertMap,cameraModel);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}

#define BACKPROJECT_GPU_EXPLICIT_INSTANTIATION(type,camera)                  \
    template void backproject(const DeviceTensor2<float> &,                  \
                              DeviceTensor2<Eigen::UnalignedVec3<float> > &, \
                              camera##CameraModel<type> &)

ALL_CAMERAS_AND_TYPES_INSTANTIATION(BACKPROJECT_GPU_EXPLICIT_INSTANTIATION);

} // namespace df
