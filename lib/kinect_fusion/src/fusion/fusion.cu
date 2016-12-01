#include <df/fusion/fusion.h>

#include <Eigen/Core>

#include <df/camera/poly3.h> // TODO

#include <df/transform/rigid.h>
#include <df/transform/nonrigid.h>
#include <df/util/cudaHelpers.h>
#include <df/util/dualQuaternion.h> // TODO
#include <df/voxel/tsdf.h>
#include <df/voxel/voxelGrid.h>

//TODO
#include <stdio.h>

namespace df {

template <typename Scalar,
          typename VoxelGridT,
          typename TransformerT,
          typename CameraModelT,
          typename DepthT>
__global__ void fuseFrameKernel(VoxelGridT voxelGrid,
                                typename TransformerT::DeviceModule transformer,
                                CameraModelT cameraModel,
                                DeviceTensor2<DepthT> depthMap,
                                const Scalar truncationDistance) {

    typedef typename VoxelGridT::VoxelT_ VoxelT;
    typedef Eigen::Matrix<Scalar,3,1> Vec3;
    typedef Eigen::Matrix<Scalar,2,1> Vec2;
    typedef Eigen::Matrix<int,3,1> Vec3i;
    typedef Eigen::Matrix<int,2,1> Vec2i;

    static constexpr Scalar border = Scalar(5);
    static constexpr Scalar maxWeight = Scalar(100);

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    // TODO
    for (int z = threadIdx.z; z < voxelGrid.size(2); z += blockDim.z) {

        const Vec3i gridCoord(x,y,z);

        const Vec3 worldCoord = voxelGrid.gridToWorld(gridCoord);

        const Vec3 liveCoord = transformer.transformWorldToLive(worldCoord);

//        printf("%f,%f,%f\n",liveCoord(0),liveCoord(1),liveCoord(2));

        if (liveCoord(2) <= 0) {
            // the point is behind the camera;
            continue;
        }

        const Vec2 liveProjection = cameraModel.project(liveCoord);

        if (!depthMap.inBounds(liveProjection,border)) {
            // the point is out-of-frame
            continue;
        }

        // TODO: use bilinear interpolation?
        // TODO: or take min of 4 surrounding depths?
        const Vec2i discretizedProjection = round(liveProjection);

        const DepthT d = depthMap(discretizedProjection);

        if (d <= DepthT(0)) {
            // no depth measurement
            continue;
        }

        const Scalar signedDistance = d - liveCoord(2);

//        printf("%f (%f - %f)\n",signedDistance,d,liveCoord(2));

        if (signedDistance < -truncationDistance) {
            // the point is too far behind the observation
            continue;
        }

        const Scalar truncatedSignedDistance = signedDistance > truncationDistance ? truncationDistance : signedDistance;

        VoxelT & voxel = voxelGrid(x,y,z);
        voxel.fuse(truncatedSignedDistance,1.f,maxWeight);

    }

}

template <typename Scalar,
          typename VoxelGridT,
          typename TransformerT,
          typename CameraModelT,
          typename DepthT>
void fuseFrame(VoxelGridT & voxelGrid,
               TransformerT & transformer,
               CameraModelT & cameraModel,
               Tensor<2,DepthT,DeviceResident> & depthMap,
               const Scalar truncationDistance) {

    dim3 block(16,16,4);
    dim3 grid(intDivideAndCeil(voxelGrid.size(0),block.x),
              intDivideAndCeil(voxelGrid.size(1),block.y),
              1);

    fuseFrameKernel<Scalar,VoxelGridT,TransformerT,CameraModelT,DepthT><<<grid,block>>>(voxelGrid,transformer.deviceModule(),cameraModel,depthMap,truncationDistance);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}

template void fuseFrame(VoxelGrid<float,TsdfVoxel,DeviceResident> &,
                        RigidTransformer<float> &,
                        Poly3CameraModel<float> &,
                        Tensor<2,float,DeviceResident> &,
                        const float);

template void fuseFrame(VoxelGrid<float,TsdfVoxel,DeviceResident> &,
                        NonrigidTransformer<float,DualQuaternion> &,
                        Poly3CameraModel<float> &,
                        Tensor<2,float,DeviceResident> &,
                        const float);

template void fuseFrame(VoxelGrid<float,TsdfVoxel,DeviceResident> &,
                        NonrigidTransformer<float,Sophus::SE3Group> &,
                        Poly3CameraModel<float> &,
                        Tensor<2,float,DeviceResident> &,
                        const float);


} // namespace df
