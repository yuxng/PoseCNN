#pragma once

#include <df/optimization/linearSystems.h>
#include <df/transform/nonrigid.h>

#include <Eigen/Sparse>

namespace df {

typedef unsigned char uchar;

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply>
void icpNonrigid(NonrigidTransformer<Scalar,TransformT> & transformer,
                 Eigen::SparseMatrix<ScalarOpt> & dataJTJ,
                 Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & dataJTr,
                 const Scalar diagonalRegularization,
                 const Scalar regularizationWeight);

template <typename Scalar, typename ScalarOpt, typename CameraModelT,
          template <typename,int...> class TransformT, int K,
          internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply,
          typename ... DebugArgTs>
void nonrigidICP(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                 DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & predictedWarpedVertices,
                 DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & predictedWarpedNormals,
                 const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > & predictedCanonicalVertices,
                 const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > & predictedCanonicalNormals,
                 const CameraModelT & cameraModel,
                 NonrigidTransformer<Scalar,TransformT> & transformer,
                 const Sophus::SE3<Scalar> & updatePredictionToLive,
                 const Eigen::Matrix<Scalar,2,1> & depthRange,
                 const uint numIterations,
                 const Scalar diagonalRegularization,
                 const Scalar regularizationWeight,
                 DebugArgTs ... debugArgs);

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT>
Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> solveSparseLinearSystem(const NonrigidTransformer<Scalar,TransformT> & transformer,
                                                                  const Eigen::SparseMatrix<ScalarOpt> & JTJ,
                                                                  const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & JTr,
                                                                  const Scalar diagonalRegularization);

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply>
void updateDeformationGraphTransforms(NonrigidTransformer<Scalar,TransformT> & transformer,
                                      const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & vectorizedUpdate);

namespace internal {

template <typename Scalar, typename ScalarOpt, typename CameraModelT,
          template <typename,int...> class TransformT, int K,
          internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply,
          typename ... DebugArgTs>
void computeDataNormalEquations(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                                const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & predictedWarpedVertices,
                                const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & predictedWarpedNormals,
                                const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > & predictedCanonicalVertices,
                                const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > & predictedCanonicalNormals,
                                const CameraModelT & cameraModel,
                                NonrigidTransformer<Scalar,TransformT> & transformer,
                                const Sophus::SE3<Scalar> & updatePredictionToLive,
                                const Eigen::Matrix<Scalar,2,1> & depthRange,
                                std::vector<Eigen::Triplet<ScalarOpt> > & JTJTriplets,
                                Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & JTr,
                                DebugArgTs ... debugArgs);


} // namespace internal

} // namespace df
