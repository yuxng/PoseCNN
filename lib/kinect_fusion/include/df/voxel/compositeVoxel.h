#pragma once

#include <tuple>

#include <cuda_runtime.h>

#include <df/util/tupleHelpers.h>

#include <cstdio> // TODO;

namespace df {

template <typename Scalar, typename ... VoxelTs>
struct CompositeVoxel {

    inline __host__ __device__
    static CompositeVoxel<Scalar,VoxelTs ...> zero() {

        return { std::make_tuple(VoxelTs::zero()...) };

    }

    inline __host__ __device__
    CompositeVoxel<Scalar,VoxelTs ...> & operator=(const CompositeVoxel<Scalar,VoxelTs...> & other) {

        static_assert(sizeof...(VoxelTs),"must contain at least one voxel");

        copy(voxels,other.voxels);
        return *this;

    }

    template <typename VoxelT>
    inline __host__ __device__ void fuse(const typename VoxelT::ObservationType thisValue,
                                         const Scalar thisWeight, const Scalar maxWeight) {

        VoxelT & voxel = getByType<VoxelT>(voxels);
        voxel.fuse(thisValue,thisWeight,maxWeight);

    }

    template <typename VoxelT>
    inline __host__ __device__ typename VoxelT::ObservationType value() const {

        return getByType<VoxelT>(voxels).value();

    }

    template <typename VoxelT>
    inline __host__ __device__ Scalar weight() const {

        return getByType<VoxelT>(voxels).weight();

    }

    std::tuple<VoxelTs ...> voxels;

};

} // namespace df
