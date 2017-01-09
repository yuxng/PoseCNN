#pragma once

#include <Eigen/Core>
//#include <Eigen/StdVector>

#include <vector>

#include <cuda_runtime.h>

namespace Eigen {

template <typename Scalar, int D>
using UnalignedVec = Eigen::Matrix<Scalar,D,1,Eigen::DontAlign>;

template <typename Scalar>
using UnalignedVec2 = Eigen::Matrix<Scalar,2,1,Eigen::DontAlign>;

template <typename Scalar>
using UnalignedVec3 = Eigen::Matrix<Scalar,3,1,Eigen::DontAlign>;

template <typename Scalar>
using UnalignedVec4 = Eigen::Matrix<Scalar,4,1,Eigen::DontAlign>;

}

namespace df {


// -=-=-=- alignment -=-=-=-
template <typename T>
using EigenAlignedVector = std::vector<T,Eigen::aligned_allocator<T> >;


// -=-=-=- round -=-=-=-
template <typename Derived>
__host__ __device__
inline Eigen::Matrix<int,Eigen::internal::traits<Derived>::RowsAtCompileTime,Eigen::internal::traits<Derived>::ColsAtCompileTime>
round(const Eigen::MatrixBase<Derived> & v) {

    return v.array().round().matrix().template cast<int>();

}

// -=-=-=- vector manipulation -=-=-=-
//template <typename Scalar, typename Derived,
//          typename std::enable_if<std::is_same<Eigen::internal::traits<Derived>::Scalar,Scalar>::value &&
//                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> compose(Scalar head, Eigen::Matrix<Eigen::internal)

//template <typename Scalar>
//__host__ __device__
//inline Eigen::Matrix<Scalar,1,1> reverse(const Eigen::Matrix<Scalar,1,1> & vec) {

//    return vec;

//}

//template <typename Scalar, int D>
//__host__ __device__
//inline Eigen::Matrix<Scalar,D,1> reverse(const Eigen::Matrix<Scalar,D,1> & vec) {

//    Eigen::Matrix<Scalar,D,1> rev;
//    rev.template head<1>() = vec.template tail<1>();
//    rev.template tail<D-1>() = reverse(vec.template head<D-1>());
//    return rev;

//}

//template <typename Derived,
//          typename std::enable_if<Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
//                                  Eigen::internal::traits<Derived>::RowsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> reverse(const Eigen::MatrixBase<Derived> & vec) {

//    return vec;

//}

//template <typename Derived,
//          typename std::enable_if<Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> reverse(const Eigen::MatrixBase<Derived> & vec) {

//    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
//    static constexpr int D = Eigen::internal::traits<Derived>::RowsAtCompileTime;

//    Eigen::Matrix<Scalar,D,1> rev;
//    rev.template head<1>() = vec.template tail<1>();
//    if ( D > 0) {
//        rev.template tail<D-1>() = reverse(vec.template head<D-1>());
//    }
//    return rev;

//}

// -=-=-=- generic comparisons -=-=-=-
template <typename Scalar, int D>
struct VecCompare {

    typedef Eigen::Matrix<Scalar,D,1> Vec;

    __host__ __device__
    static inline bool less(const Vec & a, const Vec & b) {
        if (a(0) < b(0)) return true;
        if (a(0) > b(0)) return false;

        return VecCompare<Scalar,D-1>::less(a.template tail<D-1>(),b.template tail<D-1>());
    }

    __host__ __device__
    static inline bool equal(const Vec & a, const Vec & b) {

        if (a(0) != b(0)) return false;

        return VecCompare<Scalar,D-1>::equal(a.template tail<D-1>(),b.template tail<D-1>());
    }

};

template <typename Scalar>
struct VecCompare<Scalar,1> {

    typedef Eigen::Matrix<Scalar,1,1> Vec;

    __host__ __device__
    static inline bool less(const Vec & a, const Vec & b) {

        return a(0) < b(0);

    }

    __host__ __device__
    static inline bool equal(const Vec & a, const Vec & b) {

        return a(0) == b(0);

    }

};

// as functors
template <typename Scalar, int D>
struct VecLess {

    typedef Eigen::Matrix<Scalar,D,1> Vec;

    __host__ __device__
    inline bool operator()(const Vec & a, const Vec & b) {

        return VecCompare<Scalar,D>::less(a,b);

    }

};

template <typename Scalar, int D>
struct VecEqual {

    typedef Eigen::Matrix<Scalar,D,1> Vec;

    __host__ __device__
    inline bool operator()(const Vec & a, const Vec & b) {

        return VecCompare<Scalar,D>::equal(a,b);

    }


};

} // namespace df
