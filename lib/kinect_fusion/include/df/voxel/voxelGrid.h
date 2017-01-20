#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <df/util/tensor.h>
#include <df/voxel/color.h>
#include <df/voxel/probability.h>
#include <df/voxel/tsdf.h>

namespace df {

namespace internal {

template <Residency R>
class VoxelGridInitializer {
public:

    template <typename VoxelT>
    static void initialize(Tensor<3,VoxelT,R> & grid);

};

template <Residency R>
class VoxelGridFiller {
public:

    template <typename VoxelT>
    static void fill(Tensor<3,VoxelT,R> & grid, const VoxelT & value);

};

} // namespace internal


// this struct in effect defines an interface that all
// voxel types must adhere to in order to be useable throughout
template <typename Scalar, typename VoxelT>
struct SignedDistanceValueExtractor {

    typedef Scalar ReturnType;
    typedef Scalar ScalarType;

    __host__ __device__
    inline Scalar operator()(const VoxelT & voxel) const {
        return voxel.template value<TsdfVoxel>();
    }

};

template <typename Scalar, typename VoxelT>
struct SignedDistanceValidExtractor {

    __host__ __device__
    inline bool operator()(const VoxelT & voxel) const {
        return voxel.template weight<TsdfVoxel>() > Scalar(0);
    }

};

template <typename Scalar, typename VoxelT>
struct ColorValueExtractor {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> ReturnType;
    typedef Scalar ScalarType;

    __host__ __device__
    inline ReturnType operator()(const VoxelT & voxel) const {
        return voxel.template value<ColorVoxel>();
    }

};

template <typename Scalar, typename VoxelT>
struct ProbabilityValueExtractor {

    typedef Eigen::Matrix<Scalar,10,1,Eigen::DontAlign> ReturnType;
    typedef Scalar ScalarType;

    __host__ __device__
    inline ReturnType operator()(const VoxelT & voxel) const {
        return voxel.template value<ProbabilityVoxel>();
    }

};

template <typename Scalar,
          typename VoxelT,
          Residency R>
class VoxelGrid {
public:

    typedef VoxelT VoxelT_;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<uint,3,1> Vec3ui;
    typedef Eigen::AlignedBox<Scalar,3> BoundingBox;

    VoxelGrid(Vec3ui dimensions, VoxelT * data,
              const Vec3 scale, const Vec3 offset)
        : grid_(dimensions, data), scale_(scale), offset_(offset) { }

    VoxelGrid(Vec3ui dimensions, VoxelT * data,
              const BoundingBox boundingBox)
        : grid_(dimensions, data) {

        offset_ = boundingBox.min();
        scale_ = dimensions.cast<Scalar>().unaryExpr([](Scalar x){ return Scalar(1)/(x-1); })
                .cwiseProduct(boundingBox.max() - boundingBox.min());

    }

    void fill(const VoxelT & value) {
        internal::VoxelGridFiller<R>::fill(grid_,value);
    }

    inline __host__ __device__ unsigned int size(const unsigned int dimension) const {
        return grid_.dimensionSize(dimension);
    }

    inline __host__ __device__ Vec3ui dimensions() const {
        return grid_.dimensions();
    }

    template <typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                                        Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                                        std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ Vec3 gridToWorld(const Eigen::MatrixBase<Derived> & gridCoord) const {
        return gridCoord.template cast<Scalar>().cwiseProduct(scale_) + offset_;
    }

    inline __host__ __device__ Vec3 worldToGrid(Vec3 worldCoord) const {
        return (worldCoord - offset_).cwiseProduct(
                    scale_.unaryExpr([](const Scalar val){ return Scalar(1) / val; }));
    }

    inline __host__ __device__ const Vec3 & gridToWorldOffset() const {
        return offset_;
    }

    inline __host__ __device__ const Vec3 & gridToWorldScale() const {
        return scale_;
    }

    inline __host__ __device__ const Vec3 worldToGridScale() const {
        return scale_.unaryExpr([](const Scalar val){ return Scalar(1) / val; });
    }

    inline __host__ __device__ VoxelT & operator()(const uint x, const uint y, const uint z) {
        return grid_(x,y,z);
    }

    inline __host__ __device__ const VoxelT & operator()(const uint x, const uint y, const uint z) const {
        return grid_(x,y,z);
    }

    template <typename IdxT, int Options>
    inline __host__ __device__ VoxelT & operator()(const Eigen::Matrix<IdxT,3,1,Options> & indices) {
        return grid_(indices);
    }

    template <typename IdxT, int Options>
    inline __host__ __device__ const VoxelT & operator()(const Eigen::Matrix<IdxT,3,1,Options> & indices) const {
        return grid_(indices);
    }

    inline __host__ __device__ Vec3 min() const {
        return offset_;
    }

    inline __host__ __device__ Vec3 max() const {
        const Vec3ui dims = grid_.dimensions() - Vec3ui(1,1,1);
        return min() + dims.cast<Scalar>().cwiseProduct(scale_);
    }

    inline __host__ __device__ BoundingBox boundingBox() const {
//        const Vec3 min = offset_;
//        const Eigen::Matrix<uint,3,1> dims = grid_.dimensions() - Eigen::Matrix<uint,3,1>(1,1,1);
//        const Vec3 max = min + dims.cast<Scalar>().cwiseProduct(scale_);
        return BoundingBox(min(),max());
    }

    inline __host__ __device__ Tensor<3,VoxelT,R> grid() const {
        return grid_;
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,4,4> gridToWorldTransform() const {

        Eigen::Matrix<Scalar,4,4> T = Eigen::Matrix<Scalar,4,4>::Identity();

        for (int i = 0; i < 3; ++ i) {
            T(i,i) = scale_(i);
            T(i,3) = offset_(i);
        }

        return T;

    }

private:

    Tensor<3,VoxelT,R> grid_;
    Vec3 scale_;
    Vec3 offset_;
};

template <typename Scalar, typename VoxelT>
using DeviceVoxelGrid = VoxelGrid<Scalar,VoxelT,DeviceResident>;

} // namespace df
