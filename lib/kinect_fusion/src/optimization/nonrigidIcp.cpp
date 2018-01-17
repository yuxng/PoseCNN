#include <df/optimization/nonrigidIcp.h>

#include <df/optimization/deformationGraphRegularization.h>
#include <df/util/dualQuaternion.h>
#include <df/util/typeList.h>

#include <sophus/se3.hpp>

#include <Eigen/CholmodSupport>

#include <fstream> // TODO

namespace df {

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT>
Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> solveSparseLinearSystem(const NonrigidTransformer<Scalar,TransformT> & transformer,
                                                                  const Eigen::SparseMatrix<ScalarOpt> & JTJ,
                                                                  const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & JTr,
                                                                  const Scalar diagonalRegularization) {

    typedef Eigen::SparseMatrix<ScalarOpt> SparseMat;

    static Eigen::CholmodSimplicialLDLT<SparseMat, Eigen::Upper> solver;

    static uint lastTransformerSerialNumber = std::numeric_limits<uint>::max();

    const uint currentTransformerSerialNumber = transformer.serialNumber();

//    std::cout << "summed: " << std::endl;
//    std::cout << "max: " << JTJ.toDense().maxCoeff() << std::endl;
//    std::cout << "min: " << JTJ.toDense().minCoeff() << std::endl;

//    std::cout << JTJ.rows() << " x " << JTJ.cols() << std::endl;
//    std::cout << JTr.rows() << " x " << JTr.cols() << std::endl;

//    std::cout << "max: " << JTr.maxCoeff() << std::endl;
//    std::cout << "min: " << JTr.minCoeff() << std::endl;

    for (int k=0; k<JTJ.outerSize(); ++k) {
        for (typename SparseMat::InnerIterator it(JTJ,k); it; ++it) {
//            std::cout << it.row() << ", " << it.col() << std::endl;
            if (!std::isfinite(it.value())) {
                throw std::runtime_error(std::to_string(it.row()) + "," + std::to_string(it.col()) + " not finite (" + std::to_string(it.value()) + ")");
            }
        }
    }

    for (int r = 0; r < JTr.rows(); ++r) {
        if (!std::isfinite(JTr(r))) {
            throw std::runtime_error("row " + std::to_string(r) + " not finite (" + std::to_string(JTr(r)) + ")");
        }
    }

    // check if the transformer's serial number has changed.
    // if so, the deformation graph has been updated and we should
    // redo the pattern analysis.
    if (currentTransformerSerialNumber != lastTransformerSerialNumber) {

        solver.analyzePattern(JTJ);

        lastTransformerSerialNumber = currentTransformerSerialNumber;

    }

    solver.setShift(diagonalRegularization);

    solver.factorize(JTJ);

    return -solver.solve(JTr);

}

//template Eigen::VectorXd solveSparseLinearSystem(const NonrigidTransformer<double,DualQuaternion> &,
//                                                 const Eigen::SparseMatrix<double> &,
//                                                 const Eigen::VectorXd &,
//                                                 const double);

//template Eigen::VectorXd solveSparseLinearSystem(const NonrigidTransformer<double,Sophus::SE3> &,
//                                                 const Eigen::SparseMatrix<double> &,
//                                                 const Eigen::VectorXd &,
//                                                 const double);


template <typename Scalar, int Options, template <typename,int...> class TransformT>
inline TransformT<Scalar,Options> applyUpdate(TransformT<Scalar,Options> & transform,
                                      const TransformT<Scalar,Options> & update,
                                      const IntToType<internal::TransformUpdateLeftMultiply>) {

//    std::cout << "left update" << std::endl;
    return update*transform;

}

template <typename Scalar, int Options, template <typename,int...> class TransformT>
inline TransformT<Scalar,Options> applyUpdate(TransformT<Scalar,Options> & transform,
                                      const TransformT<Scalar,Options> & update,
                                      const IntToType<internal::TransformUpdateRightMultiply>) {

//    std::cout << "right update" << std::endl;
    return transform*update;

}

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply>
void updateDeformationGraphTransforms(NonrigidTransformer<Scalar,TransformT> & transformer,
                                      const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & vectorizedUpdate) {

    typedef TransformT<Scalar> Transform;

    static constexpr uint BlockDim = 6;

    assert(vectorizedUpdate.rows() == BlockDim * transformer.numVerticesTotal());

    uint runningVertexCount = 0;

    for (uint level = 0; level < transformer.numRegularizationTreeLevels(); ++level) {

//        std::cout << "updating level " << level << std::endl;

        for (uint index = 0; index < transformer.numVerticesAtLevel(level); ++index) {

//            std::cout << "updating vertex " << index << std::endl;

//            std::cout << vectorizedUpdate.template block<6,1>((runningVertexCount + index)*BlockDim,0).transpose() << std::endl;

            // TODO: investigate an exponential mappint directly into the DualQuaternion space
            Transform update = Sophus::SE3<Scalar>::exp(vectorizedUpdate.template block<6,1>((runningVertexCount + index)*BlockDim,0).template cast<Scalar>());

            Transform & transform = transformer.transforms(level)[index];

            transform = applyUpdate(transform,update,IntToType<U>());

        }

        runningVertexCount += transformer.numVerticesAtLevel(level);

    }

}

//template void updateDeformationGraphTransforms(NonrigidTransformer<float,Sophus::SE3> &,
//                                               const Eigen::VectorXf &);

//template void updateDeformationGraphTransforms(NonrigidTransformer<float,DualQuaternion> &,
//                                               const Eigen::VectorXf &);



template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U = internal::TransformUpdateLeftMultiply>
void icpNonrigid(NonrigidTransformer<Scalar,TransformT> & transformer,
                 Eigen::SparseMatrix<ScalarOpt> & dataJTJ,
                 Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & dataJTr,
                 const Scalar diagonalRegularization,
                 const Scalar regularizationWeight) {

    Eigen::SparseMatrix<ScalarOpt> regularizerJTJ(6*transformer.numVerticesTotal(),6*transformer.numVerticesTotal());
    Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> regularizerJTr(6*transformer.numVerticesTotal());

    std::cout << "compute regularizer normal equations:" << std::endl;
    computeRegularizerNormalEquations<Scalar,ScalarOpt,TransformT,U>(transformer,regularizerJTJ,regularizerJTr);

//    std::cout << regularizerJTJ.toDense() << std::endl;

    std::cout << "solve sparse linear system" << std::endl;
    const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> solution = solveSparseLinearSystem<Scalar,ScalarOpt,TransformT>(transformer,
//                                                                                                                    dataJTJ,
//                                                                                                                    dataJTr,
                                                                                                                    regularizationWeight*regularizerJTJ + dataJTJ,
                                                                                                                    regularizationWeight*regularizerJTr + dataJTr,
//                                                                                                                    regularizerJTJ,
//                                                                                                                    regularizerJTr,
                                                                                                                    diagonalRegularization);
    std::cout << "update deformation graph transforms" << std::endl;
    updateDeformationGraphTransforms<Scalar,ScalarOpt,TransformT,U>(transformer,solution);

}


template void icpNonrigid<float,double,Sophus::SE3,internal::TransformUpdateLeftMultiply>
    (NonrigidTransformer<float,Sophus::SE3> &,
     Eigen::SparseMatrix<double> &,
     Eigen::VectorXd &, float, float);

template void icpNonrigid<float,double,Sophus::SE3,internal::TransformUpdateRightMultiply>
    (NonrigidTransformer<float,Sophus::SE3> &,
     Eigen::SparseMatrix<double> &,
     Eigen::VectorXd &, float, float);

template void icpNonrigid<float,double,DualQuaternion,internal::TransformUpdateLeftMultiply>
    (NonrigidTransformer<float,DualQuaternion> &,
     Eigen::SparseMatrix<double> &,
     Eigen::VectorXd &, float, float);

template void icpNonrigid<float,double,DualQuaternion,internal::TransformUpdateRightMultiply>
    (NonrigidTransformer<float,DualQuaternion> &,
     Eigen::SparseMatrix<double> &,
     Eigen::VectorXd &, float, float);


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
                 DebugArgTs ... debugArgs) {

    typedef Eigen::Triplet<ScalarOpt> Triplet;
    typedef Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> VecX;

    static constexpr int ModelDim = 6;

    const int numBaseLevelVertices = transformer.numVerticesAtLevel(0);

    const int numVerticesTotal = transformer.numVerticesTotal();

    for (uint iteration = 0; iteration < numIterations; ++iteration) {

        // -=-=-=- update prediction -=-=-=-
        transformer.template warpMesh<K,2,4>(predictedWarpedVertices, predictedWarpedNormals,
                                             predictedCanonicalVertices, predictedCanonicalNormals);

        // -=-=-=- data term -=-=-=-
        std::vector<Triplet> JTJTriplets;
        VecX dataJTr;

        internal::computeDataNormalEquations<Scalar,ScalarOpt,CameraModelT,TransformT,K,U,DebugArgTs...>(
                    liveVertices,
                    predictedWarpedVertices, predictedWarpedNormals,
                    predictedCanonicalVertices, predictedCanonicalNormals,
                    cameraModel, transformer, updatePredictionToLive, depthRange,
                    JTJTriplets, dataJTr, debugArgs...);

        Eigen::SparseMatrix<ScalarOpt> dataJTJ(numVerticesTotal * ModelDim, numVerticesTotal * ModelDim);

        dataJTJ.setFromTriplets(JTJTriplets.begin(), JTJTriplets.end());

        dataJTJ.finalize();

        //    std::cout << "data: " << std::endl;

        //    std::ofstream dataJTJStream("/tmp/dataJTJ.txt");
        //    dataJTJStream << dataJTJ.toDense() << std::endl;
        //    std::cout << "max: " << dataJTJ.toDense().maxCoeff() << std::endl;
        //    std::cout << "min: " << dataJTJ.toDense().minCoeff() << std::endl;

        //    std::cout << dataJTJ.coeffRef(588,588) << std::endl;

        // -=-=-=- regularization term -=-=-=-

        Eigen::SparseMatrix<ScalarOpt> regularizerJTJ(ModelDim * numVerticesTotal,ModelDim * numVerticesTotal);
        Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> regularizerJTr(ModelDim * numVerticesTotal);

        computeRegularizerNormalEquations<Scalar,ScalarOpt,TransformT,U>(transformer,regularizerJTJ,regularizerJTr);

        //    std::cout << "reg: " << std::endl;

        //    std::ofstream regJTJStream("/tmp/regJTJ.txt");
        //    regJTJStream << regularizerJTJ.toDense() << std::endl;

        //    std::cout << "max: " << regularizerJTJ.toDense().maxCoeff() << std::endl;
        //    std::cout << "min: " << regularizerJTJ.toDense().minCoeff() << std::endl;

        //    std::cout << regularizerJTJ.coeffRef(588,588) << std::endl;

        // -=-=-=- solution -=-=-=-
        Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> fullDataJTr(ModelDim * numVerticesTotal);

        fullDataJTr.head(ModelDim*numBaseLevelVertices) = dataJTr;

        fullDataJTr.tail(ModelDim*(numVerticesTotal - numBaseLevelVertices)) = Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1>::Zero( ModelDim*(numVerticesTotal - numBaseLevelVertices),1);

        const Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> solution = solveSparseLinearSystem<Scalar,ScalarOpt,TransformT>(transformer,
                                                                                                                        regularizationWeight*regularizerJTJ + dataJTJ,
                                                                                                                        regularizationWeight*regularizerJTr + fullDataJTr,
                                                                                                                        diagonalRegularization);

        // -=-=-=- solution -=-=-=-
        updateDeformationGraphTransforms<Scalar,ScalarOpt,TransformT,U>(transformer,solution);


    }

}

#define NONRIGID_ICP_EXPLICIT_INSTANTIATION(type, opt_type, camera, transform, K, update)                                       \
template void nonrigidICP<type,opt_type,camera##CameraModel<type>,transform, K, internal::TransformUpdate##update##Multiply>(   \
    const DeviceTensor2<Eigen::UnalignedVec3<type>  > &,                                                                        \
    DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                               \
    DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                               \
    const DeviceTensor2<Eigen::UnalignedVec4<type>  > &,                                                                        \
    const DeviceTensor2<Eigen::UnalignedVec4<type>  > &,                                                                        \
    const camera##CameraModel<type> &,                                                                                          \
    NonrigidTransformer<type,transform> &,                                                                                      \
    const Sophus::SE3<type> &,                                                                                             \
    const Eigen::Matrix<type,2,1> &,                                                                                            \
    const uint, const type, const type);                                                                                        \
                                                                                                                                \
template void nonrigidICP<type,opt_type,camera##CameraModel<type>,transform, K, internal::TransformUpdate##update##Multiply, DeviceTensor2<Eigen::UnalignedVec4<uchar> > >(   \
    const DeviceTensor2<Eigen::UnalignedVec3<type>  > &,                                                                        \
    DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                               \
    DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                               \
    const DeviceTensor2<Eigen::UnalignedVec4<type>  > &,                                                                        \
    const DeviceTensor2<Eigen::UnalignedVec4<type>  > &,                                                                        \
    const camera##CameraModel<type> &,                                                                                          \
    NonrigidTransformer<type,transform> &,                                                                                      \
    const Sophus::SE3<type> &,                                                                                             \
    const Eigen::Matrix<type,2,1> &,                                                                                            \
    const uint, const type, const type,                                                                                         \
    DeviceTensor2<Eigen::UnalignedVec4<uchar> > )



NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Poly3, DualQuaternion, 4, Left);
NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Poly3, DualQuaternion, 4, Right);

NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Poly3, Sophus::SE3, 4, Left);
NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Poly3, Sophus::SE3, 4, Right);


NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Linear, DualQuaternion, 4, Left);
NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Linear, DualQuaternion, 4, Right);

NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Linear, Sophus::SE3, 4, Left);
NONRIGID_ICP_EXPLICIT_INSTANTIATION(float, double, Linear, Sophus::SE3, 4, Right);

} // namespace df
