#pragma once

#include <Eigen/Core>

#include <cuda_runtime.h>

#include <df/util/cudaHelpers.h>

namespace df {

namespace internal {

enum TransformUpdateMethod {
    TransformUpdateLeftMultiply = 0,
    TransformUpdateRightMultiply
};

// -=-=-=-=- helper data structures -=-=-=-=-
template <typename Scalar, uint ResidualDim, uint ModelDim>
struct JacobianAndResidual {
    Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign> J;
    Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign> r;
};

template <typename Scalar, uint ModelDim>
struct JacobianAndResidual<Scalar,1,ModelDim> {
    Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> J;
    Scalar r;
};


template <typename Scalar, uint D>
struct UpperTriangularMatrix {
    Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor> head;
    UpperTriangularMatrix<Scalar,D-1> tail;

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,D> zero() {
        return { Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>::Zero(), UpperTriangularMatrix<Scalar,D-1>::zero() };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,D> operator+(const UpperTriangularMatrix<Scalar,D> & other) const {
        return { head + other.head, tail + other.tail };
    }
};

template <typename Scalar>
struct UpperTriangularMatrix<Scalar,1> {
    Eigen::Matrix<Scalar,1,1,Eigen::DontAlign | Eigen::RowMajor> head;

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> zero() {
        return { Eigen::Matrix<Scalar,1,1,Eigen::DontAlign | Eigen::RowMajor>::Zero() };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> operator+(const UpperTriangularMatrix & other) const {
        return { head + other.head };
    }
};


template <typename Scalar, uint ModelDim>
struct LinearSystem {

    static __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> zero() {
        return { UpperTriangularMatrix<Scalar,ModelDim>::zero(), Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor>::Zero() };
    }

    UpperTriangularMatrix<Scalar,ModelDim> JTJ;
    Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor> JTr;
};



// -=-=-=-=- helper functors -=-=-=-=-
//template <typename Scalar, uint D>
//struct StaticDotProduct {

//    static __attribute__((always_inline)) __host__ __device__
//    Scalar dot(const Eigen::Matrix<Scalar,D,1> & a, const Eigen::Matrix<Scalar,D,1> & b) {
//        return a(0)*b(0) + StaticDotProduct<Scalar,D-1>::dot(a.template tail<D-1>(),b.template tail<D-1>());
//    }

//};

//template <typename Scalar>
//struct StaticDotProduct<Scalar,1> {

//    static __attribute__((always_inline)) __host__ __device__
//    Scalar dot(const Eigen::Matrix<Scalar,1,1> & a, const Eigen::Matrix<Scalar,1,1> & b) {
//        return a(0)*b(0);
//    }

//};

template <typename Scalar, uint ResidualDim, uint ModelDim, int D>
struct JTJRowInitializer {

    static __attribute__((always_inline)) __host__ __device__
    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & row,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & J) {
        row(D) = J.template block<ResidualDim,1>(0,0).dot(J.template block<ResidualDim,1>(0,D)); // StaticDotProduct<Scalar,ResidualDim>::dot(J.template block<ResidualDim,1>(0,0),J.template block<ResidualDim,1>(0,D)); //J.template block<ResidualDim,1>(0,0).transpose()*J.template block<ResidualDim,1>(0,D);
        JTJRowInitializer<Scalar,ResidualDim,ModelDim,D-1>::initializeRow(row,J);
    }

};

// recursive base case
template <typename Scalar, uint ResidualDim, uint ModelDim>
struct JTJRowInitializer<Scalar,ResidualDim,ModelDim,-1> {

    static __attribute__((always_inline)) __host__ __device__
    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*row*/,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*J*/) { }

};

//// sepcialization for 1-dimensional residuals
//template <typename Scalar, uint ModelDim, int D>
//struct JTJRowInitializer<Scalar,1,ModelDim,D> {

//    static __attribute__((always_inline)) __host__ __device__
//    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & row,
//                       const Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & J) {
//        row(D) = J(0,0)*J(0,D);
//        JTJRowInitializer<Scalar,1,ModelDim,D-1>::initializeRow(row,J);
//    }

//};

//// recursive base case for 1-dimensional specialization
//template <typename Scalar, uint ModelDim>
//struct JTJRowInitializer<Scalar,1,ModelDim,-1> {

//    static __attribute__((always_inline)) __host__ __device__
//    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*row*/,
//                       const Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*J*/) { }

//};

template <typename Scalar, uint ResidualDim, uint ModelDim>
struct JTJInitializer {

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,ModelDim> upperTriangularJTJ(const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & jacobian) {
        Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> row;
        JTJRowInitializer<Scalar,ResidualDim,ModelDim,ModelDim-1>::initializeRow(row,jacobian);
        return { row, JTJInitializer<Scalar,ResidualDim,ModelDim-1>::upperTriangularJTJ(jacobian.template block<ResidualDim,ModelDim-1>(0,1)) };
    }

};

// recursive base case
template <typename Scalar, uint ResidualDim>
struct JTJInitializer<Scalar,ResidualDim,1> {

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> upperTriangularJTJ(const Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign> & jacobian) {
        return { jacobian.transpose()*jacobian };
    }

};

//// specialization for 1-dimensional residuals
//template <typename Scalar, uint ModelDim>
//struct JTJInitializer<Scalar,1,ModelDim> {

//    static __attribute__((always_inline)) __host__ __device__
//    UpperTriangularMatrix<Scalar,ModelDim> upperTriangularJTJ(const Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & jacobian) {
//        Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> row;
//        JTJRowInitializer<Scalar,1,ModelDim,ModelDim-1>::initializeRow(row,jacobian);
//        return { row, JTJInitializer<Scalar,ResidualDim,ModelDim-1>::upperTriangularJTJ(jacobian.template block<ResidualDim,ModelDim-1>(0,1)) };
//    }

//};


template <typename Scalar, uint ResidualDim, uint ModelDim>
struct LinearSystemCreationFunctor {

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        return { JTJInitializer<Scalar,ResidualDim,ModelDim>::upperTriangularJTJ(jacobianAndResidual.J),
                 jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

    }

};

template <typename Scalar, uint ModelDim>
struct LinearSystemSumFunctor {

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const LinearSystem<Scalar,ModelDim> & lhs,
                                             const LinearSystem<Scalar,ModelDim> & rhs) {
        return { lhs.JTJ.operator +(rhs.JTJ), lhs.JTr + rhs.JTr };
    }

};

template <typename Scalar, uint ModelDim, uint Index>
struct SquareMatrixRowInitializer {

    static constexpr uint Row = ModelDim - Index;

    __attribute__((always_inline)) __host__ __device__
    static void initialize(Eigen::Matrix<Scalar,ModelDim,ModelDim> & M,
                           const UpperTriangularMatrix<Scalar,ModelDim-Row> & upperTriangle) {

        M.template block<1,ModelDim-Row>(Row,Row) = upperTriangle.head;
        SquareMatrixRowInitializer<Scalar,ModelDim,Index-1>::initialize(M,upperTriangle.tail);

    }

};

template <typename Scalar, uint ModelDim>
struct SquareMatrixRowInitializer<Scalar,ModelDim,1> {

    static constexpr uint Row = ModelDim-1;

    __attribute__((always_inline)) __host__ __device__
    static void initialize(Eigen::Matrix<Scalar,ModelDim,ModelDim> & M,
                           const UpperTriangularMatrix<Scalar,1> & upperTriangle) {

        M.template block<1,ModelDim-Row>(Row,Row) = upperTriangle.head;

    }

};

template <typename Scalar, uint ModelDim>
struct SquareMatrixReconstructor {

    static __attribute__((always_inline)) __host__ __device__
    Eigen::Matrix<Scalar,ModelDim,ModelDim> reconstruct(const UpperTriangularMatrix<Scalar,ModelDim> & upperTriangle) {

        Eigen::Matrix<Scalar,ModelDim,ModelDim> M;
        SquareMatrixRowInitializer<Scalar,ModelDim,ModelDim>::initialize(M,upperTriangle);
        return M;

    }

};

template <typename Scalar, int D>
struct VectorAtomicAdder {

    __host__ __device__ inline static
    void atomicAdd(Scalar * destination, const Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor> & source) {

        if (source(0) != Scalar(0)) {
            Scalar val = source(0);
            atomicAddX(destination,val);
        }

        VectorAtomicAdder<Scalar,D-1>::atomicAdd(destination+1,source.template block<1,D-1>(0,1));

    }

};

template <typename Scalar>
struct VectorAtomicAdder<Scalar,0> {

    __host__ __device__ inline static
    void atomicAdd(Scalar * /*destination*/, const Eigen::Matrix<Scalar,1,0,Eigen::DontAlign | Eigen::RowMajor> & /*source*/) { }

};

template <typename Scalar, int D>
struct JTJAtomicAdder {

    __host__ __device__ inline static
    void atomicAdd(UpperTriangularMatrix<Scalar,D> & destination, const UpperTriangularMatrix<Scalar,D> & source) {

        VectorAtomicAdder<Scalar,D>::atomicAdd(destination.head.data(),source.head);

        JTJAtomicAdder<Scalar,D-1>::atomicAdd(destination.tail,source.tail);

    }

};

template <typename Scalar>
struct JTJAtomicAdder<Scalar,1> {

    __host__ __device__ inline static
    void atomicAdd(UpperTriangularMatrix<Scalar,1> & destination, const UpperTriangularMatrix<Scalar,1> & source) {

        VectorAtomicAdder<Scalar,1>::atomicAdd(destination.head.data(),source.head);

    }

};

template <typename Scalar, int D>
struct LinearSystemAtomicAdder {

    __host__ __device__ inline static
    void atomicAdd(LinearSystem<Scalar,D> & destination, const LinearSystem<Scalar,D> & source) {

        JTJAtomicAdder<Scalar,D>::atomicAdd(destination.JTJ,source.JTJ);

        VectorAtomicAdder<Scalar,D>::atomicAdd(destination.JTr.data(),source.JTr);

    }

};

// huber loss

template <typename Scalar, uint ResidualDim, uint ModelDim, int D>
struct JTJRowInitializerHuber {

    static __attribute__((always_inline)) __host__ __device__
    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & row,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & J,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & rhoDoublePrimeJ) {
        row(D) = J.template block<ResidualDim,1>(0,0).dot(rhoDoublePrimeJ.template block<ResidualDim,1>(0,D)); // StaticDotProduct<Scalar,ResidualDim>::dot(J.template block<ResidualDim,1>(0,0),J.template block<ResidualDim,1>(0,D)); //J.template block<ResidualDim,1>(0,0).transpose()*J.template block<ResidualDim,1>(0,D);
        JTJRowInitializerHuber<Scalar,ResidualDim,ModelDim,D-1>::initializeRow(row, J, rhoDoublePrimeJ);
    }

};

// recursive base case
template <typename Scalar, uint ResidualDim, uint ModelDim>
struct JTJRowInitializerHuber<Scalar,ResidualDim,ModelDim,-1> {

    static __attribute__((always_inline)) __host__ __device__
    void initializeRow(Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*row*/,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*J*/,
                       const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & /*rhoDoublePrimeJ*/) { }

};

template <typename Scalar, uint ResidualDim, uint ModelDim>
struct JTJInitializerHuber {

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,ModelDim> upperTriangularJTJ(const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & jacobian,
                                                              const Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign | Eigen::RowMajor> & rhoDoublePrimeJacobian) {
        Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> row;
        JTJRowInitializerHuber<Scalar,ResidualDim,ModelDim,ModelDim-1>::initializeRow(row, jacobian, rhoDoublePrimeJacobian);
        return { row, JTJInitializerHuber<Scalar,ResidualDim,ModelDim-1>::upperTriangularJTJ(jacobian.template block<ResidualDim,ModelDim-1>(0,1), rhoDoublePrimeJacobian.template block<ResidualDim,ModelDim-1>(0,1)) };
    }

};

template <typename Scalar, uint ResidualDim>
struct JTJInitializerHuber<Scalar,ResidualDim,1> {

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> upperTriangularJTJ(const Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign> & jacobian,
                                                       const Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign> & rhoDoublePrimeJacobian) {
        return { jacobian.transpose()*rhoDoublePrimeJacobian };
    }

};

template <typename Scalar, uint ResidualDim, uint ModelDim>
struct ResidualFunctorHuber {

    ResidualFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        const Scalar norm = jacobianAndResidual.r.norm();

        if (norm < alpha_) {

            return Scalar(0.5) * norm * norm;

        } else {

            return alpha_ * (norm - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;

};

template <typename Scalar, uint ModelDim>
struct ResidualFunctorHuber<Scalar,1,ModelDim> {

    ResidualFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        const Scalar norm = fabsf(jacobianAndResidual.r);

        if (norm < alpha_) {

            return Scalar(0.5) * norm * norm;

        } else {

            return alpha_ * (norm - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;

};

template <typename Scalar, uint ResidualDim, uint R, uint C>
struct RhoDoublePrimeInitializer {

    __attribute__((always_inline))
    __host__ __device__
    static void Initialize(Eigen::Matrix<Scalar, ResidualDim, ResidualDim, Eigen::DontAlign> & rhoDoublePrime,
                           const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & residual) {

        rhoDoublePrime(R,C) = - residual(R) * residual(C);

        RhoDoublePrimeInitializer<Scalar, ResidualDim, R, C-1>::Initialize(rhoDoublePrime, residual);

    }

};

template <typename Scalar, uint ResidualDim, uint R>
struct RhoDoublePrimeInitializer<Scalar, ResidualDim, R, R> {
    //diagonal specialization

    __attribute__((always_inline))
    __host__ __device__
    static void Initialize(Eigen::Matrix<Scalar, ResidualDim, ResidualDim, Eigen::DontAlign> & rhoDoublePrime,
                           const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & residual) {

        rhoDoublePrime(R,R) = residual(R) * residual(R);

        RhoDoublePrimeInitializer<Scalar, ResidualDim, R, R-1>::Initialize(rhoDoublePrime, residual);

    }

};

template <typename Scalar, uint ResidualDim, uint R>
struct RhoDoublePrimeInitializer<Scalar, ResidualDim, R, -1> {
    // row wrappover case

    __attribute__((always_inline))
    __host__ __device__
    static void Initialize(Eigen::Matrix<Scalar, ResidualDim, ResidualDim, Eigen::DontAlign> & rhoDoublePrime,
                           const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & residual) {

        RhoDoublePrimeInitializer<Scalar, ResidualDim, R-1, ResidualDim-1>::Initialize(rhoDoublePrime, residual);

    }

};

template <typename Scalar, uint ResidualDim, uint C>
struct RhoDoublePrimeInitializer<Scalar, ResidualDim, -1, C> {
    //base case

    __attribute__((always_inline))
    __host__ __device__
    static void Initialize(Eigen::Matrix<Scalar, ResidualDim, ResidualDim, Eigen::DontAlign> & rhoDoublePrime,
                           const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & residual) { }

};

template <typename Scalar, uint ResidualDim, uint ModelDim>
struct LinearSystemCreationFunctorHuber {

    LinearSystemCreationFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        const Scalar norm = jacobianAndResidual.r.norm();

        if (norm < alpha_) {

            return { JTJInitializer<Scalar,ResidualDim,ModelDim>::upperTriangularJTJ(jacobianAndResidual.J),
                     jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

        } else {

            Eigen::Matrix<Scalar, ResidualDim, ModelDim> jacobianTransposeRhoDoublePrime;

            // TODO: could be more efficient
            const Scalar alphaOverNormCubed = alpha_ / (norm * jacobianAndResidual.r.squaredNorm());

            Eigen::Matrix<Scalar, ResidualDim, ResidualDim, Eigen::DontAlign> rhoDoublePrime;

            {

                RhoDoublePrimeInitializer<Scalar, ResidualDim, ResidualDim-1, ResidualDim-1>::Initialize(rhoDoublePrime, jacobianAndResidual.r);

                jacobianTransposeRhoDoublePrime = alphaOverNormCubed * jacobianAndResidual.J * rhoDoublePrime;

            }

            return { JTJInitializerHuber<Scalar,ResidualDim,ModelDim>::upperTriangularJTJ(jacobianAndResidual.J, jacobianTransposeRhoDoublePrime),
                     alpha_ * jacobianAndResidual.J.transpose() * jacobianAndResidual.r / norm };

        }

    }

    Scalar alpha_;

};

template <typename Scalar, uint ModelDim>
struct LinearSystemCreationFunctorHuber<Scalar, 1, ModelDim> {

    // in the special case of ResidualDim = 1, the second derivative of the Huber norm is zero, so J^T * J is cancelled out

    LinearSystemCreationFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        const Scalar norm = fabs(jacobianAndResidual.r);

        if (norm < alpha_) {

            return { JTJInitializer<Scalar,1,ModelDim>::upperTriangularJTJ(jacobianAndResidual.J),
                     jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

        } else {

            return { UpperTriangularMatrix<Scalar, ModelDim>::zero(),
                     alpha_ * jacobianAndResidual.J.transpose() * jacobianAndResidual.r / norm };

        }

    }

    Scalar alpha_;

};

} // namespace internal






//namespace internal {

//// -=-=-=-=- helper data structures -=-=-=-=-
//template <typename Scalar, uint ModelDim>
//struct LinearSystem2 {

//    static constexpr int TriangleSize = ModelDim*(ModelDim+1)/2;

//    static __attribute__((always_inline)) __host__ __device__
//    LinearSystem2<Scalar,ModelDim> zero() {
//        return { Eigen::Matrix<Scalar,TriangleSize,1,Eigen::DontAlign>::Zero(), Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor>::Zero() };
//    }

//    Eigen::Matrix<Scalar,TriangleSize,1,Eigen::DontAlign> JTJ;
//    Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor> JTr;
//};



//// -=-=-=-=- helper functors -=-=-=-=-
//template <typename Scalar, uint ResidualDim, uint Index>
//struct JTJInitializer2 {

//    static __attribute__((always_inline)) __host__ __device__
//    void initialize(Eigen::Matrix<Scalar,Index*(Index+1)/2,1> & upperTriangle,
//                    const Eigen::Matrix<Scalar,ResidualDim,Index,Eigen::DontAlign | Eigen::RowMajor> & jacobian) {
//        JTJRowInitializer<Scalar,ResidualDim,Index,Index-1>::initializeRow(upperTriangle.template head<Index>(),jacobian);
//        JTJInitializer2<Scalar,ResidualDim,Index>::initialize(upperTriangle.template tail<(Index*(Index+1)/2)- Index>(),
//                                                              jacobian.template block<ResidualDim,Index-1>(0,1));
//    }

//};

//template <typename Scalar, uint ResidualDim>
//struct JTJInitializer2<Scalar,ResidualDim,1> {

//    static __attribute__((always_inline)) __host__ __device__
//    void initialize(Eigen::Matrix<Scalar,1,1> & upperTriangle,
//                    const Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign | Eigen::RowMajor> & jacobian) {
//        JTJRowInitializer<Scalar,ResidualDim,1,0>::initializeRow(upperTriangle,jacobian);
//    }

//};

//template <typename Scalar, uint ResidualDim, uint ModelDim>
//struct LinearSystemCreationFunctor2 {

//    __attribute__((always_inline)) __host__ __device__
//    LinearSystem2<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

//        LinearSystem2<Scalar,ModelDim> system;
//        JTJInitializer2<Scalar,ResidualDim,ModelDim>::initialize(system.JTJ,jacobianAndResidual.J);
//        system.JTr = jacobianAndResidual.J.transpose() * jacobianAndResidual.r;
//        return system;

//    }

//};

//template <typename Scalar>
//struct LinearSystemCreationFunctor2<Scalar,1,6> {

//    __attribute__((always_inline)) __host__ __device__
//    LinearSystem2<Scalar,6> operator()(const JacobianAndResidual<Scalar,1,6> & jacobianAndResidual) {

//        LinearSystem2<Scalar,6> system;
//        system.JTJ(0) = jacobianAndResidual.J(0)*jacobianAndResidual.J(0);
//        system.JTJ(1) = jacobianAndResidual.J(0)*jacobianAndResidual.J(1);
//        system.JTJ(2) = jacobianAndResidual.J(0)*jacobianAndResidual.J(2);
//        system.JTJ(3) = jacobianAndResidual.J(0)*jacobianAndResidual.J(3);
//        system.JTJ(4) = jacobianAndResidual.J(0)*jacobianAndResidual.J(4);
//        system.JTJ(5) = jacobianAndResidual.J(0)*jacobianAndResidual.J(5);

//        system.JTJ(6+0) = jacobianAndResidual.J(1)*jacobianAndResidual.J(1);
//        system.JTJ(6+1) = jacobianAndResidual.J(1)*jacobianAndResidual.J(2);
//        system.JTJ(6+2) = jacobianAndResidual.J(1)*jacobianAndResidual.J(3);
//        system.JTJ(6+3) = jacobianAndResidual.J(1)*jacobianAndResidual.J(4);
//        system.JTJ(6+4) = jacobianAndResidual.J(1)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+0) = jacobianAndResidual.J(2)*jacobianAndResidual.J(2);
//        system.JTJ(6+5+1) = jacobianAndResidual.J(2)*jacobianAndResidual.J(3);
//        system.JTJ(6+5+2) = jacobianAndResidual.J(2)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+3) = jacobianAndResidual.J(2)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+0) = jacobianAndResidual.J(3)*jacobianAndResidual.J(3);
//        system.JTJ(6+5+4+1) = jacobianAndResidual.J(3)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+4+2) = jacobianAndResidual.J(3)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+3+0) = jacobianAndResidual.J(4)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+4+3+1) = jacobianAndResidual.J(4)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+3+2+0) = jacobianAndResidual.J(5)*jacobianAndResidual.J(5);

//        system.JTr = jacobianAndResidual.J.transpose() * jacobianAndResidual.r;
//        return system;

//    }

//};


//template <typename Scalar, uint ModelDim>
//struct LinearSystemSumFunctor2 {

//    __attribute__((always_inline)) __host__ __device__
//    LinearSystem2<Scalar,ModelDim> operator()(const LinearSystem2<Scalar,ModelDim> & lhs,
//                                              const LinearSystem2<Scalar,ModelDim> & rhs) {
//        return { lhs.JTJ + rhs.JTJ, lhs.JTr + rhs.JTr };
//    }

//};

//template <typename Scalar, uint ModelDim, uint Index>
//struct SquareMatrixRowInitializer2 {

//    static constexpr uint Row = ModelDim-Index;

//    static __attribute__((always_inline)) __host__ __device__
//    void initialize(Eigen::Matrix<Scalar,ModelDim,ModelDim,Eigen::DontAlign> & M,
//                    const Eigen::Matrix<Scalar,Index*(Index+1)/2,1,Eigen::DontAlign> & upperTriangle) {

//        M.template block<1,ModelDim-Row>(Row,Row) = upperTriangle.template head<Index>();
//        SquareMatrixRowInitializer2<Scalar,ModelDim,Index-1>::initialize(M,upperTriangle.template tail<(Index*(Index+1)/2) - Index>());

//    }

//};

//template <typename Scalar, uint ModelDim>
//struct SquareMatrixRowInitializer2<Scalar,ModelDim,1> {

//    static constexpr uint Row = ModelDim-1;

//    static __attribute__((always_inline)) __host__ __device__
//    void initialize(Eigen::Matrix<Scalar,ModelDim,ModelDim,Eigen::DontAlign> & M,
//                    const Eigen::Matrix<Scalar,1,1,Eigen::DontAlign> & upperTriangle) {

//        M.template block<1,ModelDim-Row>(Row,Row) = upperTriangle;

//    }

//};

//template <typename Scalar, uint ModelDim>
//struct SquareMatrixReconstructor2 {

//    static __attribute__((always_inline)) __host__ __device__
//    Eigen::Matrix<Scalar,ModelDim,ModelDim,Eigen::DontAlign> reconstruct(const Eigen::Matrix<Scalar,ModelDim*(ModelDim+1)/2,1,Eigen::DontAlign> & upperTriangle) {

//        Eigen::Matrix<Scalar,ModelDim,ModelDim,Eigen::DontAlign> M;
//        SquareMatrixRowInitializer2<Scalar,ModelDim,ModelDim>::initialize(M,upperTriangle);
//        return M;

//    }

//};

//} // namespace internal






//namespace internal {

//template <typename Scalar, uint D>
//struct RawVec {

//    Scalar head;
//    RawVec<Scalar,D-1> tail;

//    inline __host__ __device__ RawVec<Scalar,D> operator+(const RawVec<Scalar,D> & other) const {
//        return { head + other.head, tail.operator +(other.tail) };
//    }

//    inline __host__ __device__ static RawVec<Scalar,D> zero() {
//        return { 0, RawVec<Scalar,D-1>::zero() };
//    }

//    inline __host__ __device__ Scalar & operator()(const uint index) {
//        return (&head)[index];
//    }

//};


//template <typename Scalar>
//struct RawVec<Scalar,1> {

//    Scalar head;

//    inline __host__ __device__ static RawVec<Scalar,1> zero() {
//        return { Scalar(0) };
//    }

//    inline __host__ __device__ RawVec<Scalar,1> operator+(const RawVec<Scalar,1> & other) const {
//        return { head + other.head };
//    }

//};

//template <typename Scalar, uint ModelDim>
//struct LinearSystem3 {

//    static constexpr int TriangleSize = ModelDim*(ModelDim+1)/2;

//    static __attribute__((always_inline)) __host__ __device__
//    LinearSystem3<Scalar,ModelDim> zero() {
//        return { RawVec<Scalar,TriangleSize>::zero(), Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor>::Zero() };
//    }

//    RawVec<Scalar,TriangleSize> JTJ;
//    Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor> JTr;
//};


//template <typename Scalar, uint ResidualDim, uint ModelDim>
//struct LinearSystemCreationFunctor3 {

//};


//// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

//template <typename Scalar>
//struct LinearSystemCreationFunctor3<Scalar,1,6> {

//    __attribute__((always_inline)) __host__ __device__
//    LinearSystem3<Scalar,6> operator()(const JacobianAndResidual<Scalar,1,6> & jacobianAndResidual) {

//        LinearSystem3<Scalar,6> system;
//        system.JTJ(0) = jacobianAndResidual.J(0)*jacobianAndResidual.J(0);
//        system.JTJ(1) = jacobianAndResidual.J(0)*jacobianAndResidual.J(1);
//        system.JTJ(2) = jacobianAndResidual.J(0)*jacobianAndResidual.J(2);
//        system.JTJ(3) = jacobianAndResidual.J(0)*jacobianAndResidual.J(3);
//        system.JTJ(4) = jacobianAndResidual.J(0)*jacobianAndResidual.J(4);
//        system.JTJ(5) = jacobianAndResidual.J(0)*jacobianAndResidual.J(5);

//        system.JTJ(6+0) = jacobianAndResidual.J(1)*jacobianAndResidual.J(1);
//        system.JTJ(6+1) = jacobianAndResidual.J(1)*jacobianAndResidual.J(2);
//        system.JTJ(6+2) = jacobianAndResidual.J(1)*jacobianAndResidual.J(3);
//        system.JTJ(6+3) = jacobianAndResidual.J(1)*jacobianAndResidual.J(4);
//        system.JTJ(6+4) = jacobianAndResidual.J(1)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+0) = jacobianAndResidual.J(2)*jacobianAndResidual.J(2);
//        system.JTJ(6+5+1) = jacobianAndResidual.J(2)*jacobianAndResidual.J(3);
//        system.JTJ(6+5+2) = jacobianAndResidual.J(2)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+3) = jacobianAndResidual.J(2)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+0) = jacobianAndResidual.J(3)*jacobianAndResidual.J(3);
//        system.JTJ(6+5+4+1) = jacobianAndResidual.J(3)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+4+2) = jacobianAndResidual.J(3)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+3+0) = jacobianAndResidual.J(4)*jacobianAndResidual.J(4);
//        system.JTJ(6+5+4+3+1) = jacobianAndResidual.J(4)*jacobianAndResidual.J(5);

//        system.JTJ(6+5+4+3+2+0) = jacobianAndResidual.J(5)*jacobianAndResidual.J(5);

//        system.JTr = jacobianAndResidual.J.transpose() * jacobianAndResidual.r;
//        return system;

//    }

//};

//template <typename Scalar, uint ModelDim>
//struct LinearSystemSumFunctor3 {

//    __attribute__((always_inline)) __host__ __device__
//    LinearSystem3<Scalar,ModelDim> operator()(const LinearSystem3<Scalar,ModelDim> & lhs,
//                                              const LinearSystem3<Scalar,ModelDim> & rhs) {
//        return { lhs.JTJ.operator+(rhs.JTJ), lhs.JTr + rhs.JTr };
//    }

//};

//} // namespace internal




} // namespace df
