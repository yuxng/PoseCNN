#include <df/optimization/deformationGraphRegularization.h>

#include <vector>
#include <Eigen/Core>

#include <df/util/dualQuaternion.h> // TODO
#include <df/util/typeList.h>

namespace df {

template <typename Scalar, typename ScalarOpt, int Options, template <typename,int...> class TransformT>
inline void computeEdgeResidual(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexA, const TransformT<Scalar,Options> & transformA,
                                const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexB, const TransformT<Scalar,Options> & transformB,
                                Eigen::Matrix<ScalarOpt,3,1,Eigen::DontAlign> & residual,
                                Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexADisplacementByTransformA,
                                Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexADisplacementByTransformB,
                                Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexARelativeToVertexB) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    vertexADisplacementByTransformA = transformA.translation();

    vertexARelativeToVertexB = vertexA - vertexB;

    vertexADisplacementByTransformB = transformB * vertexARelativeToVertexB;

    const Vec3 vertexAWarpedByTransformA = vertexA + vertexADisplacementByTransformA;

    const Vec3 vertexAWarpedByTransformB = vertexB + vertexADisplacementByTransformB;

    residual = (vertexAWarpedByTransformA - vertexAWarpedByTransformB).template cast<ScalarOpt>();

}

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT>
inline void fillEdgeJacobianAndResidual(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexA, const TransformT<Scalar> & transformA,
                                        const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexB, const TransformT<Scalar> & transformB,
                                        internal::JacobianAndResidual<ScalarOpt,3,12> & jacobianAndResidual,
                                        const IntToType<internal::TransformUpdateLeftMultiply> ) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    Vec3 vertexADisplacementByTransformA, vertexADisplacementByTransformB, vertexARelativeToVertexB;

    computeEdgeResidual(vertexA,transformA,vertexB,transformB,jacobianAndResidual.r,
                        vertexADisplacementByTransformA, vertexADisplacementByTransformB,
                        vertexARelativeToVertexB);

    // the first block is for vertex A
    jacobianAndResidual.J.template block<3,6>(0,0) << 1, 0, 0,  0, vertexADisplacementByTransformA(2), -vertexADisplacementByTransformA(1),
                                                      0, 1, 0,  -vertexADisplacementByTransformA(2), 0, vertexADisplacementByTransformA(0),
                                                      0, 0, 1,  vertexADisplacementByTransformA(1), -vertexADisplacementByTransformA(0), 0;

    // the second block is for vertex B
    jacobianAndResidual.J.template block<3,6>(0,6) << -1,  0,  0,  0, -vertexADisplacementByTransformB(2), vertexADisplacementByTransformB(1),
                                                       0, -1,  0,  vertexADisplacementByTransformB(2), 0, -vertexADisplacementByTransformB(0),
                                                       0,  0, -1,  -vertexADisplacementByTransformB(1), vertexADisplacementByTransformB(0), 0;

}

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT>
inline void fillEdgeJacobianAndResidual(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexA, const TransformT<Scalar> & transformA,
                                        const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertexB, const TransformT<Scalar> & transformB,
                                        internal::JacobianAndResidual<ScalarOpt,3,12> & jacobianAndResidual,
                                        const IntToType<internal::TransformUpdateRightMultiply> ) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<ScalarOpt,3,3,Eigen::DontAlign> Mat3;

    Vec3 vertexADisplacementByTransformA, vertexADisplacementByTransformB, vertexARelativeToVertexB;

    computeEdgeResidual(vertexA,transformA,vertexB,transformB,jacobianAndResidual.r,
                        vertexADisplacementByTransformA, vertexADisplacementByTransformB,
                        vertexARelativeToVertexB);

    // the first block is for vertex A
    jacobianAndResidual.J.template block<3,3>(0,0) = transformA.rotationMatrix().template cast<ScalarOpt>();
    jacobianAndResidual.J.template block<3,3>(0,3) = Mat3::Zero();

    // the second block is for vertex B
    jacobianAndResidual.J.template block<3,3>(0,6) = -transformB.rotationMatrix().template cast<ScalarOpt>();

    Mat3 partial;
    partial << 0, vertexARelativeToVertexB(2), -vertexARelativeToVertexB(1),
               -vertexARelativeToVertexB(2), 0, vertexARelativeToVertexB(0),
               vertexARelativeToVertexB(1), -vertexARelativeToVertexB(0), 0;

    jacobianAndResidual.J.template block<3,3>(0,9) = -transformB.rotationMatrix().template cast<ScalarOpt>()*partial;

}

template <typename Scalar, uint Rows>
struct JacobianFillColUnroller {

    typedef Eigen::Triplet<Scalar> Triplet;

    inline static void fill(std::vector<Triplet> & globalJacobianTriplets,
                            const Eigen::Matrix<Scalar,Rows,1,Eigen::DontAlign> & localJacobianColumn,
                            const uint firstRowIndex, const uint columnIndex) {

//        std::cout << "colFill " << firstRowIndex << ", " << columnIndex << ":" << std::endl << localJacobianColumn << std::endl << std::endl;

        const Scalar & localVal = localJacobianColumn(0);

        if (localVal != Scalar(0)) {

            globalJacobianTriplets.push_back(Triplet(firstRowIndex,columnIndex,localVal));

        }

        JacobianFillColUnroller<Scalar,Rows-1>::fill(globalJacobianTriplets,
                                                     localJacobianColumn.template tail<Rows-1>(),
                                                     firstRowIndex+1,columnIndex);

    }

};

template <typename Scalar>
struct JacobianFillColUnroller<Scalar,0> {

    typedef Eigen::Triplet<Scalar> Triplet;

    inline static void fill(std::vector<Triplet> & /*globalJacobianTriplets*/,
                            const Eigen::Matrix<Scalar,0,1,Eigen::DontAlign> & /*localJacobianColumn*/,
                            const uint /*firstRowIndex*/, const uint /*columnIndex*/) { }

};

template <typename Scalar, uint Rows, uint Cols>
struct JacobianFillUnroller {

    typedef Eigen::Triplet<Scalar> Triplet;

    inline static void fill(std::vector<Triplet> & globalJacobianTriplets,
                            const Eigen::Matrix<Scalar,Rows,Cols,Eigen::DontAlign> & localJacobian,
                            const uint firstRowIndex, const uint firstColumnIndex) {

//        std::cout << "matFill " << firstRowIndex << ", " << firstColumnIndex << ":" << std::endl << localJacobian << std::endl << std::endl;

        JacobianFillColUnroller<Scalar,Rows>::fill(globalJacobianTriplets,
                                                   localJacobian.template block<Rows,1>(0,0),
                                                   firstRowIndex, firstColumnIndex);

        JacobianFillUnroller<Scalar,Rows,Cols-1>::fill(globalJacobianTriplets,
                                                       localJacobian.template block<Rows,Cols-1>(0,1),
                                                       firstRowIndex, firstColumnIndex + 1);

    }

};


template <typename Scalar, uint Rows>
struct JacobianFillUnroller<Scalar,Rows,0> {

    typedef Eigen::Triplet<Scalar> Triplet;

    inline static void fill(std::vector<Triplet> & /*globalJacobianTriplets*/,
                            const Eigen::Matrix<Scalar,Rows,0,Eigen::DontAlign> & /*localJacobian*/,
                            const uint /*firstRowIndex*/, const uint /*firstColumnIndex*/) { }

};

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U>
void fillRegularizationResidualAndJacobianTripletList(std::vector<Eigen::Triplet<ScalarOpt> > & globalJacobianTripletList,
                                                      Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & globalResidual,
                                                      const NonrigidTransformer<Scalar,TransformT> & transformer) {

    typedef typename NonrigidTransformer<Scalar,TransformT>::Transform Transform;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    static constexpr uint ResidualDim = 3;
    static constexpr uint BlockDim = 6;

    uint runningRowCounter = 0;

    uint lowerLevelParameterStart = 0;
    uint higherLevelParameterStart;

    for (uint level = 0; level < transformer.numRegularizationTreeLevels() - 1; ++level) {

        const uint numLowerLevelVertices = transformer.numVerticesAtLevel(level);

        higherLevelParameterStart = lowerLevelParameterStart + numLowerLevelVertices*BlockDim;

        for (uint lowerLevelIndex = 0; lowerLevelIndex < numLowerLevelVertices; ++lowerLevelIndex) {

            const Vec3 & lowerLevelVertex = transformer.deformationGraphVertices(level)[lowerLevelIndex];

            const Transform & lowerLevelTransform = transformer.transforms(level)[lowerLevelIndex];

            const uint numNeighbors = transformer.numHigherLevelNeighbors(level,lowerLevelIndex);

            for (uint k = 0; k < numNeighbors; ++k) {

                const uint higherLevelIndex = transformer.higherLevelNeighbors(level,lowerLevelIndex)[k];

                const Vec3 & higherLevelVertex = transformer.deformationGraphVertices(level+1)[higherLevelIndex];

                const Transform & higherLevelTransform = transformer.transforms(level+1)[higherLevelIndex];

                internal::JacobianAndResidual<ScalarOpt,ResidualDim,2*BlockDim> jacobianAndResidual;

                fillEdgeJacobianAndResidual(lowerLevelVertex,lowerLevelTransform,
                                            higherLevelVertex,higherLevelTransform,
                                            jacobianAndResidual,IntToType<U>());

//                std::cout << level << ", " << lowerLevelIndex << ", " << k << ": " << std::endl << jacobianAndResidual.J << std::endl << std::endl;

//                std::vector<Eigen::Triplet<ScalarOpt> > localTripletList;
//                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(localTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,0),
//                                                                           0, lowerLevelParameterStart + lowerLevelIndex*BlockDim);

//                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(localTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,BlockDim),
//                                                                           0, higherLevelParameterStart + higherLevelIndex*BlockDim);
//                Eigen::SparseMatrix<ScalarOpt> localJacobian(3,transformer.numVerticesTotal()*BlockDim);
//                localJacobian.setFromTriplets(localTripletList.begin(),localTripletList.end());
//                localJacobian.finalize();
//                std::cout << localJacobian.toDense() << std::endl << std::endl;

                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(globalJacobianTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,0),
                                                                           runningRowCounter, lowerLevelParameterStart + lowerLevelIndex*BlockDim);

                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(globalJacobianTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,BlockDim),
                                                                           runningRowCounter, higherLevelParameterStart + higherLevelIndex*BlockDim);

//                std::cout << "(" << runningRowCounter << " - " << (runningRowCounter + ResidualDim) << ") / " << globalResidual.rows() << std::endl;
                globalResidual.template block<ResidualDim,1>(runningRowCounter,0) = jacobianAndResidual.r;

                runningRowCounter += ResidualDim;


                fillEdgeJacobianAndResidual(higherLevelVertex,higherLevelTransform,
                                            lowerLevelVertex,lowerLevelTransform,
                                            jacobianAndResidual,IntToType<U>());

//                std::cout << level << ", " << lowerLevelIndex << ", " << k << ": " << std::endl << jacobianAndResidual.J << std::endl << std::endl;


                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(globalJacobianTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,BlockDim),
                                                                           runningRowCounter, lowerLevelParameterStart + lowerLevelIndex*BlockDim);

                JacobianFillUnroller<ScalarOpt,ResidualDim,BlockDim>::fill(globalJacobianTripletList,jacobianAndResidual.J.template block<ResidualDim,BlockDim>(0,0),
                                                                           runningRowCounter, higherLevelParameterStart + higherLevelIndex*BlockDim);

//                std::cout << "(" << runningRowCounter << " - " << (runningRowCounter + ResidualDim) << ") / " << globalResidual.rows() << std::endl;
                globalResidual.template block<ResidualDim,1>(runningRowCounter,0) = jacobianAndResidual.r;

                runningRowCounter += ResidualDim;

            }

        }

        lowerLevelParameterStart = higherLevelParameterStart;

    }

}

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U>
void computeRegularizerNormalEquations(const NonrigidTransformer<Scalar,TransformT> & transformer,
                                       Eigen::SparseMatrix<ScalarOpt> & JTJ,
                                       Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & JTr) {

    // TODO: come up with a nice way to compute only the upper triangle of JTJ,
    // as that is the only part used in the solver anyway

    typedef Eigen::Triplet<ScalarOpt> Triplet;
    typedef Eigen::SparseMatrix<ScalarOpt> SparseMatX;
    typedef Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> VecX;

    static constexpr uint ResidualDim = 3;
    static constexpr uint BlockDim = 6;
    // TODO TODO TODO:
    static constexpr uint NumNeighbors = 4;

    const uint numTotalDeformationGraphVertices = transformer.numVerticesTotal();

    const uint numTopLevelVertices = transformer.numVerticesAtLevel(transformer.numRegularizationTreeLevels()-1);

    const uint totalResidualRows = 2 * (numTotalDeformationGraphVertices - numTopLevelVertices) * ResidualDim * NumNeighbors;

    std::vector<Triplet> globalJacobianTriplets;
    globalJacobianTriplets.reserve(totalResidualRows*2*BlockDim);

    VecX globalResidual(totalResidualRows);

    fillRegularizationResidualAndJacobianTripletList<Scalar,ScalarOpt,TransformT,U>(globalJacobianTriplets,globalResidual,transformer);

    const uint totalModelDimension = numTotalDeformationGraphVertices * BlockDim;

    SparseMatX globalJacobian(totalResidualRows,totalModelDimension);

    globalJacobian.setFromTriplets(globalJacobianTriplets.begin(),globalJacobianTriplets.end());

    globalJacobian.finalize();

    JTJ = globalJacobian.transpose()*globalJacobian;

    JTr = globalJacobian.transpose()*globalResidual;
}

template void computeRegularizerNormalEquations<float,double,DualQuaternion,internal::TransformUpdateLeftMultiply>
                                                     (const NonrigidTransformer<float,DualQuaternion> &,
                                                      Eigen::SparseMatrix<double> &,
                                                      Eigen::VectorXd &);

template void computeRegularizerNormalEquations<float,double,DualQuaternion,internal::TransformUpdateRightMultiply>
                                                     (const NonrigidTransformer<float,DualQuaternion> &,
                                                      Eigen::SparseMatrix<double> &,
                                                      Eigen::VectorXd &);

template void computeRegularizerNormalEquations<float,double,Sophus::SE3,internal::TransformUpdateLeftMultiply>
                                                     (const NonrigidTransformer<float,Sophus::SE3> &,
                                                      Eigen::SparseMatrix<double> &,
                                                      Eigen::VectorXd &);

template void computeRegularizerNormalEquations<float,double,Sophus::SE3,internal::TransformUpdateRightMultiply>
                                                     (const NonrigidTransformer<float,Sophus::SE3> &,
                                                      Eigen::SparseMatrix<double> &,
                                                      Eigen::VectorXd &);




} // namespace df
