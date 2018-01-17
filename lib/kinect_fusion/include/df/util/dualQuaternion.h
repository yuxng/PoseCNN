#pragma once

#include <Eigen/Core>

#include <sophus/se3.hpp>

#include <cuda_runtime.h>

namespace df {

// some definitions of operators for quaternions that are not defined by Eigen.
// these are in a separate nested namespace so the operators can be used without
// also using all of the parent namespace.
namespace operators {

template <typename Scalar, int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> operator+(const Eigen::Quaternion<Scalar,Options> & lhs, const Eigen::Quaternion<Scalar,Options> & rhs) {
    return Eigen::Quaternion<Scalar,Options>(lhs.w() + rhs.w(), lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
}

template <typename Scalar, int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> operator-(const Eigen::Quaternion<Scalar,Options> & lhs, const Eigen::Quaternion<Scalar,Options> & rhs) {
    return Eigen::Quaternion<Scalar,Options>(lhs.w() - rhs.w(), lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
}

template <typename Scalar, int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> operator*(const Scalar scalar, const Eigen::Quaternion<Scalar,Options> & rhs) {
    return Eigen::Quaternion<Scalar,Options>(scalar*rhs.w(),scalar*rhs.x(),scalar*rhs.y(),scalar*rhs.z());
}

template <typename Scalar, int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> operator/(const Eigen::Quaternion<Scalar,Options> & numerator, const Scalar denominator) {
    return (Scalar(1)/denominator)*numerator;
}

template <typename Scalar, int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> & operator*=(Eigen::Quaternion<Scalar,Options> & lhs, const Scalar rhs) {
    lhs = rhs*lhs;
    return lhs;
}

template <typename Scalar,int Options>
__host__ __device__
inline Eigen::Quaternion<Scalar,Options> operator*(const Eigen::Quaternion<Scalar,Options> & lhs, const Eigen::Quaternion<Scalar,Options> & rhs) {
    return Eigen::Quaternion<Scalar,Options>(lhs.w()*rhs.w(),
                                             lhs.w()*rhs.x() + lhs.x()*rhs.w() + lhs.y()*rhs.z() - lhs.z()*rhs.y(),
                                             lhs.w()*rhs.y() + lhs.y()*rhs.w() + lhs.z()*rhs.x() - lhs.x()*rhs.z(),
                                             lhs.w()*rhs.z() + lhs.z()*rhs.w() + lhs.x()*rhs.y() - lhs.y()*rhs.x());
}

} // namespace operators

template <typename Scalar, int Options = Eigen::AutoAlign>
class DualQuaternion {
public:

    typedef Eigen::Quaternion<Scalar,Options> Quaternion;
    typedef Eigen::Matrix<Scalar,3,1> Vec3;
    typedef Eigen::Matrix<Scalar,3,3> Mat3;

    __host__ __device__
    inline DualQuaternion(const Quaternion & nondual = Quaternion(1,0,0,0), const Quaternion & dual = Quaternion(0,0,0,0))
        : nondual_(nondual), dual_(dual) { }

    template <int OtherOptions>
    __host__ __device__
    inline DualQuaternion(const Sophus::SE3<Scalar,OtherOptions> & transform) {

        *this = transform;

    }

    template <int OtherOptions>
    __host__ __device__
    inline DualQuaternion(const DualQuaternion<Scalar,OtherOptions> & other)
        : nondual_(other.nondual()), dual_(other.dual()) { }

    __host__ __device__
    static inline DualQuaternion<Scalar,Options> identity() {

        return DualQuaternion<Scalar,Options>(Quaternion(1,0,0,0),Quaternion(0,0,0,0));

    }

    template <int OtherOptions>
    __host__ __device__
    inline DualQuaternion<Scalar,Options> & operator=(const Sophus::SE3<Scalar,OtherOptions> & transform) {

        // TODO: this could probably be simplified for speed
        *this = DualQuaternion(Quaternion(1,0,0,0),
                               Quaternion(0,transform.translation()(0)/2,transform.translation()(1)/2,transform.translation()(2)/2)) *
                DualQuaternion(Quaternion(transform.rotationMatrix()),Quaternion(0,0,0,0));

        return *this;

    }

    template <typename ReturnScalar>
    __host__ __device__
    inline DualQuaternion<ReturnScalar> cast() const {
        return DualQuaternion<ReturnScalar>(nondual_.cast<ReturnScalar>(),dual_.cast<ReturnScalar>());
    }

    __host__ __device__
    inline void normalizePartial() {

        using namespace operators;

        const Scalar normalizingFactor = Scalar(1)/nondual_.norm();
        nondual_ *= normalizingFactor;
        dual_ *= normalizingFactor;
    }

    __host__ __device__
    inline void normalize() {

        using namespace operators;

        normalizePartial();
        const Scalar dualNondualDot = nondual_.dot(dual_);
        dual_ = dual_ - dualNondualDot*nondual_;
    }

    __host__ __device__
    inline const Quaternion & nondual() const {
        return nondual_;
    }

    __host__ __device__
    inline const Quaternion & dual() const {
        return dual_;
    }

    __host__ __device__
    inline Vec3 operator*(const Vec3 & point) const {
        return point + Scalar(2)*nondual().vec().cross(nondual().vec().cross(point) + nondual().w()*point) +
                       Scalar(2)*(nondual().w()*dual().vec() - dual().w()*nondual().vec() + nondual().vec().cross(dual().vec()));
    }

    __host__ __device__
    inline DualQuaternion<Scalar,Options> operator*(const DualQuaternion<Scalar,Options> & rhs) const {

        return DualQuaternion<Scalar,Options>(nondual()*rhs.nondual(),
                                              operators::operator +(nondual()*rhs.dual(), dual()*rhs.nondual()));

    }

    __host__ __device__
    inline DualQuaternion<Scalar,Options> & operator+=(const DualQuaternion<Scalar,Options> & rhs) {

        using namespace operators;

        *this = *this + rhs;
        return *this;

    }

    __host__ __device__
    inline DualQuaternion<Scalar,Options> & operator-=(const DualQuaternion<Scalar,Options> & rhs) {

        using namespace operators;

        *this = *this - rhs;
        return *this;

    }

    __host__ __device__
    inline DualQuaternion<Scalar,Options> & operator/=(const Scalar & rhs) {

        *this = *this / rhs;
        return *this;

    }

    __host__ __device__
    inline Vec3 rotate(const Vec3 & vector) const {
        return vector + Scalar(2)*nondual_.vec().cross(nondual_.vec().cross(vector) + nondual_.w()*vector);
    }

    __host__ __device__
    inline Vec3 translation() const {
        // TODO: there is probably a more efficient version
//        return operator*(Vec3(0,0,0));
        using namespace operators;

        auto qTranslation = ( Scalar(2)*dual_ ) * nondual_.conjugate();

        return qTranslation.vec();

    }

    __host__ __device__
    inline Mat3 rotationMatrix() const {
        return nondual_.toRotationMatrix();
    }

    __host__ __device__
    inline operator Sophus::SE3<Scalar>() const {

        return Sophus::SE3<Scalar>(nondual_, translation() );

    }

private:

    Quaternion nondual_;
    Quaternion dual_;

};


// some definitions of operators for dual quaternions.
// these are in a separate nested namespace so the operators can be used without
// also using all of the parent namespace.
namespace operators {

template <typename Scalar, int Options>
__host__ __device__
inline DualQuaternion<Scalar,Options> operator*(const Scalar scalar, const DualQuaternion<Scalar,Options> & dq) {

    return dq*scalar;

}

template <typename Scalar, int Options>
__host__ __device__
inline DualQuaternion<Scalar,Options> operator*(const DualQuaternion<Scalar,Options> & dq, const Scalar scalar) {

    return DualQuaternion<Scalar,Options>(scalar*dq.nondual(),scalar*dq.dual());

}

template <typename Scalar, int Options>
__host__ __device__
inline DualQuaternion<Scalar,Options> operator/(const DualQuaternion<Scalar,Options> & dq, const Scalar scalar) {

    return DualQuaternion<Scalar,Options>(dq.nondual()/scalar,dq.dual()/scalar);

}

//template <typename Scalar>
//__host__ __device__
//inline DualQuaternion<Scalar> operator*(const DualQuaternion<Scalar> & lhs, const DualQuaternion<Scalar> & rhs) {

//    return DualQuaternion<Scalar>(lhs.nondual()*rhs.nondual(),
//                                  lhs.nondual()*rhs.dual() + lhs.dual()*rhs.nondual());

//}

template <typename Scalar, int Options>
__host__ __device__
inline DualQuaternion<Scalar,Options> operator+(const DualQuaternion<Scalar,Options> & lhs, const DualQuaternion<Scalar,Options> & rhs) {

    return DualQuaternion<Scalar,Options>(lhs.nondual() + rhs.nondual(),
                                          lhs.dual() + rhs.dual());

}

template <typename Scalar, int Options>
__host__ __device__
inline DualQuaternion<Scalar, Options> operator-(const DualQuaternion<Scalar,Options> & lhs, const DualQuaternion<Scalar,Options> & rhs) {

    return DualQuaternion<Scalar,Options>(lhs.nondual() - rhs.nondual(),
                                          lhs.dual() - rhs.dual());

}

template <typename Scalar, int Options>
inline static std::ostream & operator<<(std::ostream & stream, const Eigen::Quaternion<Scalar,Options> q) {

    stream << "[" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]";
    return stream;

}

template <typename Scalar, int Options>
inline static std::ostream & operator<<(std::ostream & stream, const DualQuaternion<Scalar,Options> dq) {

    stream << dq.nondual() << dq.dual();
    return stream;

}

} // namespace operators


} // namespace df
