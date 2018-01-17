#pragma once

#include <df/util/tensor.h>

#include <df/voxel/color.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/tsdf.h>
#include <df/voxel/voxelGrid.h>
#include <df/voxel/probability.h>

#include <sophus/se3.hpp>

namespace df {

namespace internal {

enum FusionFrame {
    DepthFrame,
    ColorFrame
};

template <typename VoxelT>
struct FusionTypeTraits;

template <>
struct FusionTypeTraits<TsdfVoxel> {

    template <typename Scalar>
    struct PackedInput {

        PackedInput(Scalar truncationDistance, DeviceTensor2<Scalar> depthImage)
            : truncationDistance(truncationDistance), depthImage(depthImage) { }

        Scalar truncationDistance;
        DeviceTensor2<Scalar> depthImage;
    };

    static constexpr FusionFrame frame = DepthFrame;

};

template <>
struct FusionTypeTraits<ColorVoxel> {

    template <typename Scalar>
    struct PackedInput {

        PackedInput(const Scalar truncationDistance, const Scalar maxWeight,
                    const DeviceTensor2<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > colorImage)
            : truncationDistance(truncationDistance), maxWeight(maxWeight), colorImage(colorImage) { }

        Scalar truncationDistance;
        Scalar maxWeight;
        DeviceTensor2<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > colorImage;
    };

    static constexpr FusionFrame frame = ColorFrame;

};


template <>
struct FusionTypeTraits<ProbabilityVoxel> {

    template <typename Scalar>
    struct PackedInput {

        PackedInput(const Scalar truncationDistance, const Scalar maxWeight,
                    const DeviceTensor2<Eigen::Matrix<Scalar,10,1,Eigen::DontAlign> > colorImage)
            : truncationDistance(truncationDistance), maxWeight(maxWeight), colorImage(colorImage) { }

        Scalar truncationDistance;
        Scalar maxWeight;
        DeviceTensor2<Eigen::Matrix<Scalar,10,1,Eigen::DontAlign> > colorImage;
    };

    static constexpr FusionFrame frame = ColorFrame;

};


//template <typename ... VoxelTs>
//struct VoxelTsToInputTs;

//template <typename VoxelT>
//struct VoxelTsToInputTs<VoxelT> {

//    template <typename Scalar>
//    using Type = typename FusionTypeTraits<VoxelT>::template PackedInput<Scalar>;

//};

//template <typename HeadVoxelT, typename ... TailVoxelTs>
//struct VoxelTsToInputTs<HeadVoxelT, TailVoxelTs...> {

//    template <typename Scalar>
//    using HeadTuple = std::tuple<typename FusionTypeTraits<HeadVoxelT>::template PackedInput<Scalar> >;

//    template <typename Scalar>
//    using TailTuple = typename VoxelTsToInputTs<TailVoxelTs...>::template Type<Scalar>;

//    template <typename Scalar>
//    using Type = decltype(std::tuple_cat(std::declval<HeadTuple<Scalar> >(),std::declval<TailTuple<Scalar> >()));

//};


} // namespace internal


//template <typename Scalar,
//          typename TransformerT,
//          typename CameraModelT,
//          typename DepthT,
//          typename ... NonTsdfVoxelTs>
//void fuseFrame(DeviceVoxelGrid<Scalar,CompositeVoxel<Scalar,TsdfVoxel,NonTsdfVoxelTs...> > & voxelGrid,
//               TransformerT & transformer,
//               CameraModelT & cameraModel,
//               Tensor<2,DepthT,DeviceResident> & depthMap,
//               const Scalar truncationDistance,
//               typename internal::VoxelTsToInputTs<NonTsdfVoxelTs...>::template Type<Scalar> & nonTsdfInput);


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
               typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput);

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
               typename internal::FusionTypeTraits<NonTsdfVoxelTs>::template PackedInput<Scalar> ... nonTsdfInput);

} // namespace df
