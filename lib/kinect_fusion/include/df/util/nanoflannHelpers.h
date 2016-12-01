#pragma once

#include <Eigen/Core>

#include <df/util/tensor.h>

#include <nanoflann.hpp>

namespace df {

template <typename Scalar>
class KDPointCloud {
public:

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    KDPointCloud(const HostTensor1<Vec3> & tensor)
        : tensor_(tensor) {
    }

    inline std::size_t kdtree_get_point_count() const {
        return tensor_.length();
    }

    inline Scalar kdtree_distance(const Scalar * p1, const std::size_t idx_p2, std::size_t /*size*/) const {
        Eigen::Map<const Vec3> v1(p1);
        const Vec3 & v2 = tensor_(idx_p2);
        return (v1-v2).squaredNorm();
    }

    inline Scalar kdtree_get_pt(const std::size_t idx, int dim) const {
        return tensor_(idx).data()[dim];
    }

    template <typename BoundingBox>
    bool kdtree_get_bbox(BoundingBox & /*bb*/) const {
        return false;
    }

private:

    const HostTensor1<Vec3> & tensor_;

};

template <typename Scalar>
using KDTree = nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<Scalar, KDPointCloud<Scalar> >, KDPointCloud<Scalar>, 3, int>;

} // namespace df
