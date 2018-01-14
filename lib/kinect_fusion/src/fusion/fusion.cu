#include <df/fusion/fusion.h>

#include <Eigen/Core>

#include <df/camera/poly3.h> // TODO

#include <df/transform/rigid.h>
#include <df/transform/nonrigid.h>
#include <df/util/cudaHelpers.h>
#include <df/util/dualQuaternion.h> // TODO
#include <df/voxel/color.h>
#include <df/voxel/probability.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/tsdf.h>
#include <df/voxel/voxelGrid.h>

//TODO
#include <stdio.h>

namespace df {

template <typename ... NonTsdfVoxelTs>
struct RequiresColorFrameFusion;

template <>
struct RequiresColorFrameFusion<> {

    static constexpr bool Value = false;

};

template <typename HeadVoxelT, typename ... TailVoxelTs>
struct RequiresColorFrameFusion<HeadVoxelT,TailVoxelTs...> {

    static constexpr bool Value = internal::FusionTypeTraits<HeadVoxelT>::frame == internal::ColorFrame ||
            RequiresColorFrameFusion<TailVoxelTs...>::Value;

};

template <typename Scalar,typename ... NonTsdfVoxelTs>
struct NoColorFusionHandler {

    __device__ inline void doColorFusion(CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> &,
                                         const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> &,
                                         const Scalar,
                                         typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ...) const { }

};

template <typename Scalar, typename VoxelT, typename CompositeVoxelT>
struct SingleColorFusionHandler {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;

    __device__
    inline void doFusion(DeviceVoxelGrid<Scalar,CompositeVoxelT> & voxel,
                         const Vec2 colorFrameProjection,
                         const Scalar signedDistance,
                         const typename internal::FusionTypeTraits<VoxelT>::template PackedInput<Scalar> & voxelInput) {



    }

};

template <typename Scalar, typename CompositeVoxelT>
struct SingleColorFusionHandler<Scalar,ColorVoxel, CompositeVoxelT> {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    __device__
    static inline void doFusion(CompositeVoxelT & voxel,
                                const Vec2 colorFrameProjection,
                                const Scalar signedDistance,
                                const typename internal::FusionTypeTraits<ColorVoxel>::template PackedInput<Scalar> & voxelInput) {

        if (fabs(signedDistance) < voxelInput.truncationDistance) {

            const DeviceTensor2<Vec3> & colorImage = voxelInput.colorImage;

            if (voxelInput.colorImage.inBounds(colorFrameProjection(0),colorFrameProjection(1),Scalar(2))) {

                const Vec3 color = colorImage.interpolate(colorFrameProjection(0),colorFrameProjection(1));

                voxel.template fuse<ColorVoxel>(color,1.f,voxelInput.maxWeight);

            }

        }

    }
};

template <typename Scalar, typename CompositeVoxelT>
struct SingleColorFusionHandler<Scalar, ProbabilityVoxel, CompositeVoxelT> {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;
    typedef Eigen::Matrix<Scalar,10,1,Eigen::DontAlign> Vec;

    __device__
    static inline void doFusion(CompositeVoxelT & voxel,
                                const Vec2 colorFrameProjection,
                                const Scalar signedDistance,
                                const typename internal::FusionTypeTraits<ProbabilityVoxel>::template PackedInput<Scalar> & voxelInput) {

        if (fabs(signedDistance) < voxelInput.truncationDistance) {

            const DeviceTensor2<Vec> & colorImage = voxelInput.colorImage;

            if (voxelInput.colorImage.inBounds(colorFrameProjection(0),colorFrameProjection(1),Scalar(2))) {

                const Vec color = colorImage.interpolate(colorFrameProjection(0),colorFrameProjection(1));

                voxel.template fuse<ProbabilityVoxel>(color,1.f,voxelInput.maxWeight);

            }

        }

    }
};

template <typename Scalar, typename CompositeVoxelT, typename ... NonTsdfVoxelTs>
struct ColorFusionForLoop;

template <typename Scalar, typename CompositeVoxelT, typename HeadVoxelT, typename ... TailVoxelTs>
struct ColorFusionForLoop<Scalar,CompositeVoxelT,HeadVoxelT,TailVoxelTs...> {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;

    __device__
    static inline void doFusion(CompositeVoxelT & voxel,
                                const Vec2 & colorFrameProjection,
                                const Scalar signedDistance,
                                typename internal::FusionTypeTraits<HeadVoxelT>::PackedInput<Scalar> headInput,
                                typename internal::FusionTypeTraits<TailVoxelTs>::template PackedInput<Scalar> ... tailInputs) {

        SingleColorFusionHandler<Scalar,HeadVoxelT,CompositeVoxelT>::doFusion(voxel,colorFrameProjection,signedDistance,headInput);

        ColorFusionForLoop<Scalar,CompositeVoxelT,TailVoxelTs...>::doFusion(voxel,colorFrameProjection,signedDistance,tailInputs...);

    }

};

template <typename Scalar, typename CompositeVoxelT>
struct ColorFusionForLoop<Scalar,CompositeVoxelT> {

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;

    __device__
    static inline void doFusion(CompositeVoxelT & voxel,
                                const Vec2 & colorFrameProjection,
                                const Scalar signedDistance) { }

};

template <typename Scalar, typename ColorCameraModelT, typename ... NonTsdfVoxelTs>
struct ColorFusionHandler {

    const Sophus::SE3<Scalar> T_cd;
    const ColorCameraModelT colorCameraModel;

    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    __device__ inline void doColorFusion(CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> & voxel,
                                         const Vec3 & liveDepthCoord,
                                         const Scalar signedDistance,
                                         typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput) const {

        const Vec3 liveColorCoord = T_cd * liveDepthCoord;

        const Vec2 projectedColorCoord = colorCameraModel.project(liveColorCoord);

        ColorFusionForLoop<Scalar,CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...>,NonTsdfVoxelTs...>::doFusion(voxel,
                                                                                                                  projectedColorCoord,
                                                                                                                  signedDistance,
                                                                                                                  nonTsdfInput...);

    }

};


template <typename Scalar,
          typename TransformerT,
          typename DepthCameraModelT,
          typename DepthT,
          typename ColorFusionHandlerT,
          typename ... NonTsdfVoxelTs>
__global__ void fuseFrameKernel(DeviceVoxelGrid<Scalar,CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> > voxelGrid,
                                const typename TransformerT::DeviceModule transformer,
                                const DepthCameraModelT depthCameraModel,
                                const Scalar truncationDistance,
                                const DeviceTensor2<DepthT> depthMap,
                                const ColorFusionHandlerT colorFusionHandler,
                                typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;
    typedef Eigen::Matrix<int,3,1,Eigen::DontAlign> Vec3i;
    typedef Eigen::Matrix<int,2,1,Eigen::DontAlign> Vec2i;

    static constexpr Scalar border = Scalar(5);
    static constexpr Scalar maxWeight = Scalar(50);

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

        const Vec2 liveProjection = depthCameraModel.project(liveCoord);

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

        if (signedDistance < -truncationDistance) {
            // the point is too far behind the observation
            continue;
        }

        const Scalar truncatedSignedDistance = signedDistance > truncationDistance ? truncationDistance : signedDistance;

        CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> & voxel = voxelGrid(x,y,z);
        voxel.template fuse<TsdfVoxel>(truncatedSignedDistance,1.f,maxWeight);

        colorFusionHandler.doColorFusion(voxel,liveCoord,signedDistance,nonTsdfInput...);

    }

}

template <typename Scalar,
          typename TransformerT,
          typename CameraModelT,
          typename DepthT,
          typename ... NonTsdfVoxelTs>
void fuseFrame(DeviceVoxelGrid<Scalar,CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> > & voxelGrid,
               const TransformerT & transformer,
               const CameraModelT & cameraModel,
               const DeviceTensor2<DepthT> & depthMap,
               const Scalar truncationDistance,
               typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput) {

    dim3 block(16,16,4);
    dim3 grid(intDivideAndCeil(voxelGrid.size(0),block.x),
              intDivideAndCeil(voxelGrid.size(1),block.y),
              1);

    static_assert(!RequiresColorFrameFusion<NonTsdfVoxelTs...>::Value,
                  "this function is for fusion into depth frame only");

    fuseFrameKernel<Scalar,TransformerT,CameraModelT,DepthT,NoColorFusionHandler<Scalar,NonTsdfVoxelTs...>,NonTsdfVoxelTs...><<<grid,block>>>
           (voxelGrid,transformer.deviceModule(),cameraModel,
            truncationDistance,depthMap, NoColorFusionHandler<Scalar,NonTsdfVoxelTs...>(), nonTsdfInput ...);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}

template <typename Scalar,
          typename TransformerT,
          typename DepthCameraModelT,
          typename ColorCameraModelT,
          typename DepthT,
          typename ... NonTsdfVoxelTs>
void fuseFrame(DeviceVoxelGrid<Scalar,CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> > & voxelGrid,
               const TransformerT & transformer,
               const DepthCameraModelT & depthCameraModel,
               const ColorCameraModelT & colorCameraModel,
               const Sophus::SE3<Scalar> & T_cd,
               const DeviceTensor2<DepthT> & depthMap,
               const Scalar truncationDistance,
               typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput) {

    dim3 block(16,16,4);
    dim3 grid(intDivideAndCeil(voxelGrid.size(0),block.x),
              intDivideAndCeil(voxelGrid.size(1),block.y),
              1);

    static_assert(RequiresColorFrameFusion<NonTsdfVoxelTs...>::Value,
                  "this function is for fusion into both depth and color frame");

    fuseFrameKernel<Scalar,TransformerT,DepthCameraModelT,DepthT,ColorFusionHandler<Scalar,ColorCameraModelT,NonTsdfVoxelTs...>,NonTsdfVoxelTs...><<<grid,block>>>
            (voxelGrid,transformer.deviceModule(),depthCameraModel,
             truncationDistance,depthMap, { T_cd, colorCameraModel }, nonTsdfInput ...);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel> > &,
                        const RigidTransformer<float> &,
                        const Poly3CameraModel<float> &,
                        const DeviceTensor2<float> &,
                        const float);

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel> > &,
                        const NonrigidTransformer<float,DualQuaternion> &,
                        const Poly3CameraModel<float> &,
                        const DeviceTensor2<float> &,
                        const float);

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel> > &,
                        const NonrigidTransformer<float,Sophus::SE3> &,
                        const Poly3CameraModel<float> &,
                        const DeviceTensor2<float> &,
                        const float);

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> > &,
                        const RigidTransformer<float> &,
                        const Poly3CameraModel<float> &,
                        const Poly3CameraModel<float> &,
                        const Sophus::SE3f &,
                        const DeviceTensor2<float> &,
                        const float,
                        typename internal::FusionTypeTraits<ColorVoxel>::PackedInput<float> );

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> > &,
                        const NonrigidTransformer<float,DualQuaternion> &,
                        const Poly3CameraModel<float> &,
                        const Poly3CameraModel<float> &,
                        const Sophus::SE3f &,
                        const DeviceTensor2<float> &,
                        const float,
                        typename internal::FusionTypeTraits<ColorVoxel>::PackedInput<float>);

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> > &,
                        const NonrigidTransformer<float,Sophus::SE3> &,
                        const Poly3CameraModel<float> &,
                        const Poly3CameraModel<float> &,
                        const Sophus::SE3f &,
                        const DeviceTensor2<float> &,
                        const float,
                        typename internal::FusionTypeTraits<ColorVoxel>::PackedInput<float>);

template void fuseFrame(DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> > &,
                        const RigidTransformer<float> &,
                        const Poly3CameraModel<float> &,
                        const Poly3CameraModel<float> &,
                        const Sophus::SE3f &,
                        const DeviceTensor2<float> &,
                        const float,
                        typename internal::FusionTypeTraits<ProbabilityVoxel>::PackedInput<float> );


} // namespace df
