#include <df/optimization/nonrigidIcp.h>

#include <df/camera/poly3.h> // TODO
#include <df/camera/linear.h> // TODO

#include <df/util/dualQuaternion.h> // TODO
#include <sophus/se3.hpp>

#include <df/util/cudaHelpers.h>
#include <df/util/debugHelpers.h>

namespace df {

//template <typename Scalar, template <typename, int...> class TransformT,
//          typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
//                                                    Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
//                                                    std::is_same<typename Eigen::internal::traits<Derived>::Scalar, Scalar>::value, int>::type = 0>
//__host__ __device__ Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> rotate(const TransformT<Scalar> & transform, const Eigen::MatrixBase<Derived> & vector);


template <typename Scalar,
          typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                                    Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                                    std::is_same<typename Eigen::internal::traits<Derived>::Scalar, Scalar>::value, int>::type = 0>
__host__ __device__ Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> rotate(const Sophus::SE3<Scalar> & transform, const Eigen::MatrixBase<Derived> & vector) {

    return transform.so3() * vector;

}

template <typename Scalar,
          typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                                    Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                                    std::is_same<typename Eigen::internal::traits<Derived>::Scalar, Scalar>::value, int>::type = 0>
__host__ __device__ Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> rotate(const DualQuaternion<Scalar> & transform, const Eigen::MatrixBase<Derived> & vector) {

    return transform.rotate(vector);

}

template <typename Scalar, typename CameraModelT, int K, template <typename, int...> class TransformT, typename ... DebugArgTs>
__global__ void computeDataNormalEquationsKernel(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > liveVertices,
                                                 const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > predictedWarpedVertices,
                                                 const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > predictedWarpedNormals,
                                                 const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > predictedCanonicalVertices,
                                                 const DeviceTensor2<Eigen::UnalignedVec4<Scalar> > predictedCanonicalNormals,
                                                 const CameraModelT cameraModel,
                                                 const Sophus::SE3<Scalar> updatePredictionToLive,
                                                 const VoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> nearestNeighborGrid,
                                                 const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > deformationGraphVertices,
                                                 const DeviceTensor1<TransformT<Scalar> > deformationGraphTransforms,
                                                 const Scalar oneOverBlendingSigmaSquared,
                                                 DeviceTensor1<internal::UpperTriangularMatrix<Scalar,6> > diagonalJTJBlocks,
                                                 DeviceTensor1<Scalar> JTr,
                                                 DeviceTensor1<int> associationCounts,
                                                 DebugArgTs ... debugArgs) {

    // TODO: add normal disagreement check

    typedef Eigen::Matrix<Scalar,4,1,Eigen::DontAlign> Vec4;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<int,3,1> Vec3i;
    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;
    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
    typedef TransformT<Scalar> Transform;

    static constexpr int border = 2; // TODO
    static constexpr Scalar rayNormDotThreshold = Scalar(0.1); // TODO
    static constexpr Scalar maxAssiciationDistance3D = Scalar(0.02); // TODO

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < predictedCanonicalVertices.dimensionSize(0) && y < predictedCanonicalVertices.dimensionSize(1)) {

        //PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,0,0,0),debugArgs...);

        const Vec4 & canonicalVertexInGridCoords = predictedCanonicalVertices(x,y);

        // ensure value is valid
        if (!isfinite(canonicalVertexInGridCoords(0))) { // TODO: use + or - inf so we can use isinf instead (which doesn't need to check against NaN)

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,255,0,255),debugArgs...);
            return;

        }

        const Vec3i nearestNeighborCanonicalVoxel = round(canonicalVertexInGridCoords.template head<3>());

        // make sure the canonical vertice is in bounds
        if (!nearestNeighborGrid.grid().inBounds(nearestNeighborCanonicalVoxel,0)) {

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,0,255,255),debugArgs...);
            return;

        }

        const Vec3 predictedWarpedVertex = updatePredictionToLive * predictedWarpedVertices(x,y).template head<3>();

        const Vec3 predictedWarpedNormal = updatePredictionToLive.so3() * predictedWarpedNormals(x,y).template head<3>();

        const Vec2 projectedWarpedVertex = cameraModel.project(predictedWarpedVertex);

        Eigen::Vector2i nearestDiscretePixel = round(projectedWarpedVertex);

        // ensure it projects in bounds
        if (!liveVertices.inBounds(nearestDiscretePixel,border)) {

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,0,0,255),debugArgs...);
            return;

        }

        // TODO: eigen vector accessor
        const Vec3 & liveVertex = liveVertices(nearestDiscretePixel);

        // make sure the point projects on valid depth
        if (liveVertex(2) <= 0 ) {

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,255,0,255),debugArgs...);
            return;

        }

        // ensure the predicted and live points are close enough
        if ( (liveVertex - predictedWarpedVertex).norm() > maxAssiciationDistance3D ) {

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,255,255,255),debugArgs...);
            return;

        }

        const Vec3 ray = predictedWarpedVertex.normalized();

        // make sure the view is not too oblique
        if ( -ray.dot(predictedWarpedNormal) < rayNormDotThreshold ) {

            PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,0,255,255),debugArgs...);
            return;

        }

        const NNVec & nearestNeighborIndices = nearestNeighborGrid(nearestNeighborCanonicalVoxel);

        const Vec3 canonicalVertexInWorldCoords = nearestNeighborGrid.gridToWorld(canonicalVertexInGridCoords.template head<3>());

        const Vec4 & canonicalNormal = predictedCanonicalNormals(x,y);

        Scalar totalResidual(0);

        // TODO: unroll?
        for (int k = 0; k < K; ++k) {

            const int neighborIndex = nearestNeighborIndices(k);

            if (neighborIndex < 0) {

                break;

            }


            const Vec3 & deformationGraphVertex = deformationGraphVertices(neighborIndex);

            const Scalar distanceSquared = (deformationGraphVertex - canonicalVertexInWorldCoords).squaredNorm();

            const Scalar weight = expX(-distanceSquared*oneOverBlendingSigmaSquared);

            const Scalar weightSquared = weight * weight;

            const Transform & deformationGraphTransform = deformationGraphTransforms(neighborIndex);

            const Vec3 canonicalVertexOffsetFromNeighbor = deformationGraphTransform*(canonicalVertexInWorldCoords - deformationGraphVertex);

            const Vec3 canonicalVertexWarpedByNeighbor = deformationGraphVertex + canonicalVertexOffsetFromNeighbor;

            const Vec3 canonicalNormalWarpedByNeighbor = rotate(deformationGraphTransform, canonicalNormal.template head<3>());

//            if (k == 0) {

//                printf("%f,%f,%f vs %f, %f, %f\n", deformationGraphVertex(0),deformationGraphVertex(1),deformationGraphVertex(2),
//                       canonicalVertexInWorldCoords(0),canonicalVertexInWorldCoords(1),canonicalVertexInWorldCoords(2));

//            }

            // TODO: apply update?
            const Vec3 residual = canonicalVertexWarpedByNeighbor - liveVertex;

            const Scalar pointPlaneResidual = canonicalNormalWarpedByNeighbor.dot(residual);

            totalResidual += pointPlaneResidual*pointPlaneResidual;

            Eigen::Matrix<Scalar,3,6> dOffset_dNeighborTransformUpdate;
            dOffset_dNeighborTransformUpdate << 1, 0, 0,  0, canonicalVertexOffsetFromNeighbor(2), -canonicalVertexOffsetFromNeighbor(1),
                                                0, 1, 0,  -canonicalVertexOffsetFromNeighbor(2), 0, canonicalVertexOffsetFromNeighbor(0),
                                                0, 0, 1,  canonicalVertexOffsetFromNeighbor(1), -canonicalVertexOffsetFromNeighbor(0), 0;

            const Eigen::Matrix<Scalar,1,6> dError_dNeighborTransformUpdate = canonicalNormalWarpedByNeighbor.template cast<Scalar>().transpose()*dOffset_dNeighborTransformUpdate;

            const internal::UpperTriangularMatrix<Scalar,6> localNeighborJTJBlock = internal::JTJInitializer<Scalar,1,6>::upperTriangularJTJ(weight * dError_dNeighborTransformUpdate);

            internal::UpperTriangularMatrix<Scalar,6> & globalNeighborJTJBlock = diagonalJTJBlocks(neighborIndex);

            internal::JTJAtomicAdder<Scalar,6>::atomicAdd(globalNeighborJTJBlock,localNeighborJTJBlock);

            internal::VectorAtomicAdder<Scalar,6>::atomicAdd(JTr.data() + 6 * neighborIndex, weightSquared * pointPlaneResidual * dError_dNeighborTransformUpdate);

            atomicAdd(&associationCounts(neighborIndex),1);

//            if (neighborIndex == 98 && x == 276 && y == 232) {

////                printf("%d,%d -> %f %f %f\n",x,y,canonicalVertexOffsetFromNeighbor(0),canonicalVertexOffsetFromNeighbor(1),canonicalVertexOffsetFromNeighbor(2));

//                printf("%d,%d -> %f\n",x,y,localNeighborJTJBlock.head(0));
//                printf("warped vertex: %f,%f,%f\n", canonicalVertexWarpedByNeighbor(0), canonicalVertexWarpedByNeighbor(1), canonicalVertexWarpedByNeighbor(2));
//                printf("predicted normal: %f,%f,%f\n", predictedWarpedNormal(0), predictedWarpedNormal(1), predictedWarpedNormal(2));
////                printf("warped normal: %f,%f,%f\n", canonicalNormalWarpedByNeighbor(0), canonicalNormalWarpedByNeighbor(1), canonicalNormalWarpedByNeighbor(2));

//            }

        }

//        printf("%f,%f,%f -> %f, %f, %f\n", canonicalVertexInGridCoords(0),canonicalVertexInGridCoords(1),canonicalVertexInGridCoords(2),
//               canonicalVertexInWorldCoords(0),canonicalVertexInWorldCoords(1),canonicalVertexInWorldCoords(2));

        const uchar gray = min(Scalar(255),255 * totalResidual / (0.01*0.01) );
        PixelDebugger<DebugArgTs...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(gray,gray,gray,255),debugArgs...);

//        printf("%f\n",totalResidual);

    }

}

namespace internal {

template <typename Scalar, typename ScalarOpt, typename CameraModelT,
          template <typename,int...> class TransformT, int K,
          internal::TransformUpdateMethod U, typename ... DebugArgTs>
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
                                DebugArgTs ... debugArgs) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Triplet<ScalarOpt> Triplet;

    static constexpr int ModelDim = 6;

    static constexpr int minimumAssociationCount = 30; // TODO

    const uint predictionWidth = predictedCanonicalVertices.dimensionSize(0);

    const uint predictionHeight = predictedCanonicalVertices.dimensionSize(1);

    assert(predictedWarpedVertices.dimensionSize(0) == predictionWidth);
    assert(predictedWarpedVertices.dimensionSize(1) == predictionHeight);
    assert(predictedWarpedNormals.dimensionSize(0) == predictionWidth);
    assert(predictedWarpedNormals.dimensionSize(1) == predictionHeight);

    const dim3 block(32,32,1); //TODO
    const dim3 grid(intDivideAndCeil(predictionWidth,block.x),intDivideAndCeil(predictionHeight,block.y),1);

    const int numBaseLevelVertices = transformer.numVerticesAtLevel(0);


    // TODO: transformer already stores this
    ManagedDeviceTensor1<Vec3> baseLevelDeformationGraphVertices(numBaseLevelVertices);
    baseLevelDeformationGraphVertices.copyFrom(ConstHostTensor1<Vec3>(numBaseLevelVertices,transformer.deformationGraphVertices(0)));

    ManagedDeviceTensor1<TransformT<Scalar> > baseLevelDeformationGraphTransforms(numBaseLevelVertices);
    baseLevelDeformationGraphTransforms.copyFrom(ConstHostTensor1<TransformT<Scalar> >(numBaseLevelVertices,transformer.transforms(0)));

    ManagedDeviceTensor1<internal::UpperTriangularMatrix<Scalar,ModelDim> > diagonalJTJBlocks(numBaseLevelVertices);
    cudaMemset(diagonalJTJBlocks.data(),0,diagonalJTJBlocks.dimensionSize(0)*sizeof(internal::UpperTriangularMatrix<Scalar,ModelDim>));

    ManagedDeviceTensor1<Scalar> deviceJTr(numBaseLevelVertices * ModelDim);
    cudaMemset(deviceJTr.data(),0,deviceJTr.dimensionSize(0)*sizeof(Scalar));

    ManagedDeviceTensor1<int> deviceAssociationCounts(numBaseLevelVertices);
    cudaMemset(deviceAssociationCounts.data(),0,deviceAssociationCounts.dimensionSize(0)*sizeof(int));

    const Scalar blendingSigma = transformer.blendingSigma();

    computeDataNormalEquationsKernel<<<grid,block>>>(liveVertices,
                                                     predictedWarpedVertices,
                                                     predictedWarpedNormals,
                                                     predictedCanonicalVertices,
                                                     predictedCanonicalNormals,
                                                     cameraModel,
                                                     updatePredictionToLive,
                                                     transformer.nearestNeighborGrid(),
                                                     baseLevelDeformationGraphVertices,
                                                     baseLevelDeformationGraphTransforms,
                                                     Scalar(1)/(blendingSigma*blendingSigma),
                                                     diagonalJTJBlocks,
                                                     deviceJTr,
                                                     deviceAssociationCounts,
                                                     debugArgs...);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

    Eigen::Matrix<Scalar,Eigen::Dynamic,1> hostJTr(numBaseLevelVertices * ModelDim);

    HostTensor1<Scalar>(numBaseLevelVertices * ModelDim, hostJTr.data()).copyFrom(deviceJTr);

    JTr = hostJTr.template cast<ScalarOpt>();

    ManagedHostTensor1<int> hostAssociationCounts(numBaseLevelVertices);
    hostAssociationCounts.copyFrom(deviceAssociationCounts);

    ManagedHostTensor1<internal::UpperTriangularMatrix<Scalar,ModelDim> > hostDiagonalJTJBlocks(diagonalJTJBlocks.dimensions());
    hostDiagonalJTJBlocks.copyFrom(diagonalJTJBlocks);

    JTJTriplets.reserve(ModelDim*ModelDim*numBaseLevelVertices);

    for (int index = 0; index < numBaseLevelVertices; ++index) {

        // make sure we have enough points to support an update to this node
        if (hostAssociationCounts(index) > minimumAssociationCount) {

            internal::UpperTriangularMatrix<Scalar,ModelDim> & upperTriangle = hostDiagonalJTJBlocks(index);

            const Eigen::Matrix<Scalar,ModelDim,ModelDim> squareMatrix = internal::SquareMatrixReconstructor<Scalar,ModelDim>::reconstruct(upperTriangle);

//            std::cout << "block " << index << ": " << std::endl;
//            std::cout << squareMatrix << std::endl << std::endl;

            for (int r = 0; r < ModelDim; ++r) {

                for (int c = r; c < ModelDim; ++c) {

                    const Scalar & val = squareMatrix(r,c);

                    if ( val != Scalar(0) ) {

                        JTJTriplets.push_back(Triplet(index * ModelDim + r, index * ModelDim + c, val));

                        // TODO
                        if ( r != c) {
                            JTJTriplets.push_back(Triplet(index * ModelDim + c, index * ModelDim + r, val));
                        }

                    }

                }

            }

        }

    }


}

#define COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(type,type_opt,camera,transform,K,update)                                           \
    template void computeDataNormalEquations<type,type_opt,camera##CameraModel<type>,transform,K,internal::TransformUpdate##update##Multiply>(  \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec4<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec4<type> > &,                                                                                     \
        const camera##CameraModel<type> &,                                                                                                      \
        NonrigidTransformer<type,transform> &,                                                                                                  \
        const Sophus::SE3<type> &,                                                                                                         \
        const Eigen::Matrix<type,2,1> &,                                                                                                        \
        std::vector<Eigen::Triplet<type_opt> > &,                                                                                               \
        Eigen::Matrix<type_opt,Eigen::Dynamic,1> &);                                                                                            \
                                                                                                                                                \
    template void computeDataNormalEquations<type,type_opt,camera##CameraModel<type>,transform,K,internal::TransformUpdate##update##Multiply, DeviceTensor2<Eigen::UnalignedVec4<uchar> > >(  \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec3<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec4<type> > &,                                                                                     \
        const DeviceTensor2<Eigen::UnalignedVec4<type> > &,                                                                                     \
        const camera##CameraModel<type> &,                                                                                                      \
        NonrigidTransformer<type,transform> &,                                                                                                  \
        const Sophus::SE3<type> &,                                                                                                         \
        const Eigen::Matrix<type,2,1> &,                                                                                                        \
        std::vector<Eigen::Triplet<type_opt> > &,                                                                                               \
        Eigen::Matrix<type_opt,Eigen::Dynamic,1> &,                                                                                             \
        DeviceTensor2<Eigen::UnalignedVec4<uchar> > )

COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Poly3,DualQuaternion,4,Left);
COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Poly3,DualQuaternion,4,Right);

COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Poly3,Sophus::SE3,4,Left);
COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Poly3,Sophus::SE3,4,Right);


COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Linear,DualQuaternion,4,Left);
COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Linear,DualQuaternion,4,Right);

COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Linear,Sophus::SE3,4,Left);
COMPUTE_DATA_NORMAL_EQUATIONS_EXPLICIT_INSTANTIATION(float,double,Linear,Sophus::SE3,4,Right);


} // namespace internal


} // namespace df
