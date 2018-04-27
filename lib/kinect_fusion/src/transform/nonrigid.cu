#include <df/transform/nonrigid.h>

#include <df/surface/decimation.h>
#include <df/util/cudaHelpers.h>
#include <df/util/dualQuaternion.h> // TODO
#include <df/util/sophusHelpers.h> // TODO

#include <thrust/device_vector.h>

namespace df {

using namespace operators;


template <int DTensor>
struct ThreadIndexer;

template <>
struct ThreadIndexer<1> {

    inline __device__ static uint getIndex(const uint3 threadIdx, const dim3 blockDim, const uint3 blockIdx ) {

        return threadIdx.x + blockDim.x * blockIdx.x;

    }

    template <typename Scalar>
    inline static std::pair<dim3,dim3> makeGridBlock(DeviceTensor1<Scalar> & input) {

        const dim3 block(1024,1,1);
        const dim3 grid(intDivideAndCeil((uint)input.length(),block.x),1,1);

        return std::pair<dim3,dim3>(grid,block);

    }

};

template <>
struct ThreadIndexer<2> {

    inline __device__ static Eigen::Matrix<uint,2,1> getIndex(const uint3 threadIdx, const dim3 blockDim, const uint3 blockIdx ) {

        return Eigen::Matrix<uint,2,1>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);

    }

    template <typename Scalar>
    inline static std::pair<dim3,dim3> makeGridBlock(DeviceTensor2<Scalar> & input) {

        const dim3 block(16,16,1);
        const dim3 grid(intDivideAndCeil(input.dimensionSize(0),block.x),
                        intDivideAndCeil(input.dimensionSize(1),block.y),1);

        return std::pair<dim3,dim3>(grid,block);

    }

};

template <typename Scalar, int K, int DTensor, int DVec>
__global__ void warpMeshKernel(DeviceTensor<DTensor, Eigen::UnalignedVec3<Scalar> > warpedVertices,
                               DeviceTensor<DTensor, Eigen::UnalignedVec3<Scalar> > warpedNormals,
                               const DeviceTensor<DTensor, Eigen::UnalignedVec<Scalar,DVec> > unwarpedVertices,
                               const DeviceTensor<DTensor, Eigen::UnalignedVec<Scalar,DVec> > unwarpedNormals,
                               const DeviceVoxelGrid<Scalar,Eigen::UnalignedVec<int,K> > nearestNeighborGrid,
                               const DeviceTensor1<Eigen::UnalignedVec3<Scalar> > deformationGraphVertices,
                               const DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > deformationGraphTransforms,
                               const Scalar oneOverBlendingSigmaSquared) {

    typedef Eigen::UnalignedVec3<Scalar> Vec3;
    typedef Eigen::UnalignedVec3<int> Vec3i;
    typedef Eigen::UnalignedVec<int,K> NNVec;
    typedef DualQuaternion<Scalar,Eigen::DontAlign> DualQuaternion;
    typedef Eigen::Quaternion<Scalar,Eigen::DontAlign> Quaternion;

    const auto i = ThreadIndexer<DTensor>::getIndex(threadIdx, blockDim, blockIdx);

    if ( warpedVertices.inBounds(i,0u) ) {

        const Vec3 unwarpedVertexGridCoords = unwarpedVertices(i).template head<3>();

        const Vec3 unwarpedNormal = unwarpedNormals(i).template head<3>();

        Vec3 & warpedVertex = warpedVertices(i);

        Vec3 & warpedNormal = warpedNormals(i);

        if (nearestNeighborGrid.grid().inBounds(unwarpedVertexGridCoords,Scalar(1))) {

            const Vec3 unwarpedVertexWorldCoords = nearestNeighborGrid.gridToWorld(unwarpedVertexGridCoords);

//            const Vec3i nearestNeighborVoxel = round(unwarpedVertexGridCoords);

//            const NNVec nearestNeighborIndices = nearestNeighborGrid(nearestNeighborVoxel);

            DualQuaternion blendedTransform(deformationGraphTransforms(0));

            const bool initialized = blendDualQuaternions(blendedTransform, unwarpedVertexGridCoords,
                                                          unwarpedVertexWorldCoords, deformationGraphVertices,
                                                          deformationGraphTransforms, nearestNeighborGrid,
                                                          oneOverBlendingSigmaSquared);

//            const Quaternion * firstNondual;

//            Scalar totalWeight;

//            bool initialized = false;

//            for (int k = 0; k < K; ++k) {

//                const int neighborIndex = nearestNeighborIndices(k);

//                if (neighborIndex < 0) {
//                    break;
//                }

//                const Vec3 & deformationGraphVertex = deformationGraphVertices(neighborIndex);

//                const DualQuaternion & deformationGraphTransform = deformationGraphTransforms(neighborIndex);

//                const Vec3 offset = unwarpedVertexWorldCoords - deformationGraphVertex;

//                const Scalar distanceSquared = offset.squaredNorm();

//                const Scalar weight = expX(-distanceSquared*oneOverBlendingSigmaSquared);


//                if (i == 100) {
//                    printf("+ %f x\n",weight);
//                    printf("%f,%f,%f,%f  %f,%f,%f,%f  (%d)\n",
//                           deformationGraphTransform.nondual().coeffs()(0),
//                           deformationGraphTransform.nondual().coeffs()(1),
//                           deformationGraphTransform.nondual().coeffs()(2),
//                           deformationGraphTransform.nondual().coeffs()(3),
//                           deformationGraphTransform.dual().coeffs()(0),
//                           deformationGraphTransform.dual().coeffs()(1),
//                           deformationGraphTransform.dual().coeffs()(2),
//                           deformationGraphTransform.dual().coeffs()(3),
//                           neighborIndex);
//                }

//                // TODO: do we need this?
//                if (weight > Scalar(0)) {

//                    if (!initialized) {

//                        blendedTransform = weight*deformationGraphTransform;

//                        totalWeight = weight;

//                        firstNondual = &deformationGraphTransform.nondual();

//                        initialized = true;

//                    } else {

//                        totalWeight += weight;

////                        if (deformationGraphTransform.nondual().dot(blendedTransform.nondual()) < Scalar(0)) {
//                        if (deformationGraphTransform.nondual().dot(*firstNondual) < Scalar(0)) {

//                            blendedTransform -= weight*deformationGraphTransform;

//                        } else {

//                            blendedTransform += weight*deformationGraphTransform;

//                        }

//                    }

//                }

//            }

            if (initialized) {

//                blendedTransform.normalize();

                warpedVertex = blendedTransform * unwarpedVertexWorldCoords;

                warpedNormal = blendedTransform.rotate(unwarpedNormal);

            } else {

                warpedVertex(0) = warpedNormal(0) = NAN;

                //warpedVertex = warpedNormal = Vec3(NAN,NAN,NAN);

            }

        } else {

            warpedVertex(0) = warpedNormal(0) = NAN;

            //warpedVertex = warpedNormal = Vec3(NAN,NAN,NAN);

        }

    }

}


template <template <typename, int...> class TransformT>
struct TransformRecenter;

template <>
struct TransformRecenter<DualQuaternion> {

    template <typename Scalar>
    inline static void recenter(DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & dualQuaternions,
                                const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
                                const ConstHostTensor1<DualQuaternion<Scalar> > & transforms) {

        using namespace operators;

        typedef Eigen::Quaternion<Scalar,Eigen::DontAlign> Quaternion;
        typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

        ManagedHostTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > recenteredDualQuaternions(dualQuaternions.dimensions());

        std::transform(transforms.data(),transforms.data() + transforms.length(), transformationOrigins.data(),
                       recenteredDualQuaternions.data(),[](const DualQuaternion<Scalar> & transform, const Vec3 & center) {

            // TODO: is there a more efficent way?
            const Quaternion decenterer(0,center(0)/2,center(1)/2,center(2)/2);

            return DualQuaternion<Scalar>(Quaternion(1,0,0,0),decenterer) * transform * DualQuaternion<Scalar>(Quaternion(1,0,0,0),Scalar(-1)*decenterer);

        });

        dualQuaternions.copyFrom(recenteredDualQuaternions);

    }

};

//template <typename Scalar>
//void NonrigidTransformer<Scalar,DualQuaternion>::toDeviceRecenteredDualQuaternions(
//        DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & dualQuaternions,
//        const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
//        const HostTensor1<DualQuaternion<Scalar> > & transforms) {

//    using namespace operators;

//    typedef Eigen::Quaternion<Scalar,Eigen::DontAlign> Quaternion;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    ManagedHostTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > recenteredDualQuaternions(dualQuaternions.dimensions());

//    std::transform(transforms.data(),transforms.data() + transforms.length(), transformationOrigins.data(),
//                   recenteredDualQuaternions.data(),[](const DualQuaternion<Scalar> & transform, const Vec3 & center) {

//        // TODO: is there a more efficent way?
//        const Quaternion centerer(0,center(0)/2,center(1)/2,center(2)/2);

//        return DualQuaternion<Scalar>(Quaternion(1,0,0,0),centerer) * transform * DualQuaternion<Scalar>(Quaternion(1,0,0,0),Scalar(-1)*centerer);

//    });

//    dualQuaternions.copyFrom(recenteredDualQuaternions);

//}

template <>
struct TransformRecenter<Sophus::SE3> {

    template <typename Scalar>
    inline static void recenter(DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & dualQuaternions,
                                const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
                                const ConstHostTensor1<Sophus::SE3<Scalar> > & transforms) {

        using namespace operators;

        typedef Eigen::Quaternion<Scalar,Eigen::DontAlign> Quaternion;
        typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

        ManagedHostTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > recenteredDualQuaternions(dualQuaternions.dimensions());

        std::transform(transforms.data(),transforms.data() + transforms.length(), transformationOrigins.data(),
                       recenteredDualQuaternions.data(),[](const DualQuaternion<Scalar> & transform, const Vec3 & center) {

            // TODO: is there a more efficent way?
            const Quaternion centerer(0,center(0)/2,center(1)/2,center(2)/2);

            return DualQuaternion<Scalar>(Quaternion(1,0,0,0),centerer) * transform * DualQuaternion<Scalar>(Quaternion(1,0,0,0),Scalar(-1)*centerer);

        });

        dualQuaternions.copyFrom(recenteredDualQuaternions);

    }

};


//template <typename Scalar>
//void NonrigidTransformer<Scalar,Sophus::SE3>::toDeviceRecenteredDualQuaternions(
//        DeviceTensor1<DualQuaternion<Scalar, Eigen::DontAlign> > & dualQuaternions,
//        const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
//        const HostTensor1<Sophus::SE3<Scalar> > & transforms) {

//    typedef Sophus::SE3<Scalar> SE3;
//    typedef DualQuaternion<Scalar,Eigen::DontAlign> DualQuaternion;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    ManagedHostTensor1<DualQuaternion> hostDualQuaternions(dualQuaternions.dimensions());

////    std::transform(transforms.data(),transforms.data() + transforms.dimensionSize(0),
////                   hostDualQuaternions.data(),[](const SE3 & transform) {  return transform; });

//    std::transform(transforms.data(),transforms.data() + transforms.length(), transformationOrigins.data(),
//                   hostDualQuaternions.data(),[](const SE3 & transform, const Vec3 & center) {

//        const Vec3 rotatedCenter = transform.so3()*center;

//        const Vec3 newTranslation = transform.translation() + center - rotatedCenter;

//        return SE3(transform.so3(),newTranslation);

//    });

//    dualQuaternions.copyFrom(hostDualQuaternions);

//}

template <typename Scalar, template <typename, int...> class TransformT>
void NonrigidTransformer<Scalar,TransformT>::toDeviceRecenteredDualQuaternions(
        DeviceTensor1<DualQuaternion<Scalar, Eigen::DontAlign> > & dualQuaternions,
        const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
        const ConstHostTensor1<TransformT<Scalar> > & transforms) const {

    TransformRecenter<TransformT>::recenter(dualQuaternions,transformationOrigins,transforms);

}

//template void NonrigidTransformer<float,Sophus::SE3>::toDeviceRecenteredDualQuaternions(
//        DeviceTensor1<DualQuaternion<float,Eigen::DontAlign> > &,
//        const ConstHostTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > &,
//        const ConstHostTensor1<Sophus::SE3<float> > &);

//template void NonrigidTransformer<float,DualQuaternion>::toDeviceRecenteredDualQuaternions(
//        DeviceTensor1<DualQuaternion<float,Eigen::DontAlign> > &,
//        const ConstHostTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > &,
//        const ConstHostTensor1<DualQuaternion<float> > &);

template <typename Scalar,template <typename,int...> class TransformT>
template <int K, int DTensor, int DVec>
void NonrigidTransformer<Scalar,TransformT>::warpMesh(DeviceTensor<DTensor,Eigen::UnalignedVec3<Scalar> > & warpedVertices,
                                                      DeviceTensor<DTensor,Eigen::UnalignedVec3<Scalar> > & warpedNormals,
                                                      const DeviceTensor<DTensor,Eigen::UnalignedVec<Scalar,DVec> > & unwarpedVertices,
                                                      const DeviceTensor<DTensor,Eigen::UnalignedVec<Scalar,DVec> > & unwarpedNormals) {

    typedef DualQuaternion<Scalar,Eigen::DontAlign> DualQuaternion;

    const std::pair<dim3,dim3> gridBlock = ThreadIndexer<DTensor>::makeGridBlock(warpedVertices);

    // TODO
    if (!invariantsHold()) {
        throw std::runtime_error("invariants don't hold");
    }

//    std::cout << "allocating dqs" << std::endl;
//    std::cout << vertexTransforms_.size() << std::endl;
//    std::cout << vertexTransforms_[0].size() << ", " << sizeof(DualQuaternion) << std::endl;
//    std::cout << deformationGraphVertices_[0].size() << std::endl;
//    std::cout << "we'll need " << vertexTransforms_[0].size() * sizeof(DualQuaternion) << " bytes" << std::endl;

//    std::cout << vertexTransforms_[0][0] << std::endl;
//    ManagedDeviceTensor1<DualQuaternion> dualQuaternionTransforms( vertexTransforms_[0].size()  );

//    toDeviceRecenteredDualQuaternions(dualQuaternionTransforms,
//                                      HostTensor1<Vec3>(deformationGraphVertices_[0].size(),deformationGraphVertices_[0].data()),
//                                      HostTensor1<Transform>(vertexTransforms_[0].size(),vertexTransforms_[0].data()));

//    ManagedHostTensor1<DualQuaternion> hostDQs( vertexTransforms_[0].size() );
//    hostDQs.copyFrom(dualQuaternionTransforms);

//    for (int i = 0; i < 10; ++i) { /*vertexTransforms_[0].size(); ++i) {*/

//        std::cout << i<< ":" << std::endl;
//        std::cout << vertexTransforms_[0][i] << std::endl;

//        std::cout << hostDQs(i) << std::endl << std::endl;

//        std::cout << sizeof(Transform) << std::endl;
//        std::cout << ((int64_t)&vertexTransforms_[0][1] - (int64_t)&vertexTransforms_[0][0]) << std::endl;
//        std::cout << std::alignment_of<Transform>::value << std::endl;

//    }

////     TODO:
//    std::cout << "allocating vertices" << std::endl;
//    ManagedDeviceTensor1<Vec3> baseLevelVertices( numVerticesAtLevel(0) );

//    std::cout << "copying vertices" << std::endl;

//    baseLevelVertices.copyFrom(ConstHostTensor1<Vec3>( numVerticesAtLevel(0) , deformationGraphVertices(0)) );

    updateDeviceVerticesAndTransforms();

    std::cout << "warping " << std::endl;

    warpMeshKernel<Scalar,4,DTensor,DVec><<<gridBlock.first,gridBlock.second>>>(
            warpedVertices,warpedNormals,
            unwarpedVertices,unwarpedNormals,
            *nearestNeighborGrid_,
            deviceBaseLevelVertices_,
            deviceBaseLevelDualQuaternionTransforms_,
            Scalar(1)/(baseDecimationResolution_*baseDecimationResolution_));

    std::cout << "warped" << std::endl;

}

template void NonrigidTransformer<float,DualQuaternion>::warpMesh<4,1,3>(DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                         DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                         const DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                         const DeviceTensor1<Eigen::UnalignedVec3<float> > &);

template void NonrigidTransformer<float,Sophus::SE3>::warpMesh<4,1,3>(DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                           DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                           const DeviceTensor1<Eigen::UnalignedVec3<float> > &,
                                                                           const DeviceTensor1<Eigen::UnalignedVec3<float> > &);

template void NonrigidTransformer<float,DualQuaternion>::warpMesh<4,2,4>(DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                                                         DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                                                         const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                                                         const DeviceTensor2<Eigen::UnalignedVec4<float> > &);

template void NonrigidTransformer<float,Sophus::SE3>::warpMesh<4,2,4>(DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                                                           DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                                                           const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                                                           const DeviceTensor2<Eigen::UnalignedVec4<float> > &);





template <typename Scalar, int K, int K2>
struct InsertionIndexSearchUnroller {

    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    __host__ __device__
    static inline void search(int & insertionIndex,
                              const VoxelGrid<Scalar,Eigen::Matrix<int,K2,1,Eigen::DontAlign>,DeviceResident> & nearestNeighborGrid,
                              const NNVec & currentNeighbors,
                              const Vec3 & voxelCenter,
                              const Scalar distance,
                              const DeviceTensor1<Vec3> & deformationGraphVertices) {

        const int neighborIndex = currentNeighbors(K-1);

        if (neighborIndex >= 0) {

            const Vec3 & neighborVertexInWorldCoords = deformationGraphVertices(neighborIndex);

            const Vec3 neighborVertexInGridCoords = nearestNeighborGrid.worldToGrid(neighborVertexInWorldCoords);

            const Vec3 neighborDiff = voxelCenter - neighborVertexInGridCoords;

            const Scalar neighborDistance = neighborDiff.squaredNorm();

            if (neighborDistance > distance) {

                insertionIndex = K-1;

            } else {

                return;

            }

        } else {

            insertionIndex = K-1;

        }

        InsertionIndexSearchUnroller<Scalar,K-1,K2>::search(insertionIndex, nearestNeighborGrid, currentNeighbors.template head<K-1>(),
                                                         voxelCenter, distance, deformationGraphVertices);

    }

};

template <typename Scalar, int K2>
struct InsertionIndexSearchUnroller<Scalar, 0, K2> {

    typedef Eigen::Matrix<int,0,1,Eigen::DontAlign> NNVec;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    __host__ __device__
    static inline void search(int & /*insertionIndex*/,
                              const VoxelGrid<Scalar,Eigen::Matrix<int,K2,1,Eigen::DontAlign>,DeviceResident> & /*nearestNeighborGrid*/,
                              const NNVec & /*currentNeighbors*/,
                              const Vec3 & /*voxelCenter*/,
                              const Scalar /*distance*/,
                              const DeviceTensor1<Vec3> & /*deformationGraphVertices*/) { }

};

// TODO: make work for anisotropic grids
template <typename Scalar, int K>
__global__ void updateDeformationGraphNearestNeighborsKernel(VoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> nearestNeighborGrid,
                                                             const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > deformationGraphVertices,
                                                             const Eigen::Matrix<int,3,1,Eigen::DontAlign> offset,
                                                             const Eigen::Matrix<int,3,1,Eigen::DontAlign> max,
                                                             const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> vertexInGridCoords, const int index) {

    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    const int x = offset(0) + threadIdx.x + blockDim.x * blockIdx.x;
    const int y = offset(1) + threadIdx.y + blockDim.y * blockIdx.y;
    const int z = offset(2) + threadIdx.z + blockDim.z * blockIdx.z;

    if (x < max(0) && y < max(1) && z < max(2)) {

//        printf("checking %d,%d,%d\n",x,y,z);

        const Vec3 diff = Vec3(x,y,z) - vertexInGridCoords;

        const Scalar distance = diff.squaredNorm();

//        printf("distance = %f\n",distance);

        NNVec & currentNeighbors = nearestNeighborGrid(x,y,z);

//        printf("current = %d %d %d %d\n",currentNeighbors(0),currentNeighbors(1),currentNeighbors(2),currentNeighbors(3));

        int insertionIndex = -1;

        InsertionIndexSearchUnroller<Scalar,K,K>::search(insertionIndex, nearestNeighborGrid, currentNeighbors,
                                                         Vec3(x,y,z), distance, deformationGraphVertices);

//        for (int k = K-1; k >= 0; --k) {

//            const int neighborIndex = currentNeighbors(k);

//            if ( neighborIndex >= 0 ) {

//                const Vec3 & neighborVertexInWorldCoords = deformationGraphVertices(neighborIndex);

//                const Vec3 neighborVertexInGridCoords = nearestNeighborGrid.worldToGrid(neighborVertexInWorldCoords);

//                const Vec3 neighborDiff = Vec3(x,y,z) - neighborVertexInGridCoords;

//                const Scalar neighborDistance = neighborDiff.squaredNorm();

//                if (neighborDistance > distance) {

//                    // inserted index is closer
//                    insertionIndex = k;

//                } else {

//                    // inserted index is farther, the search ends
//                    break;

//                }


//            } else {

//                insertionIndex = k;

//            }

//        }

        // check if the inserted vertex belongs in the updated nearest neighbor list
        if (insertionIndex >= 0) {

            for (int k = K-1; k > insertionIndex; --k) {

                currentNeighbors(k) = currentNeighbors(k-1);

            }

            currentNeighbors(insertionIndex) = index;

        }

    }

}

template <typename Scalar, int K>
void updateDeformationGraphNearestNeighbors(VoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> & nearestNeighborGrid,
                                            const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & deformationGraphVertices,
                                            const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertex, const int index,
                                            const Scalar nearestNeighborSigma, const int nSigmas = 3) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<int,3,1,Eigen::DontAlign> Vec3i;

    const Vec3 nSigmaExtent = nearestNeighborSigma * nSigmas * nearestNeighborGrid.worldToGridScale();

    const Vec3 vertexInGridCoords = nearestNeighborGrid.worldToGrid(vertex);

    const Vec3i boundingBoxMin = (vertexInGridCoords - nSigmaExtent).template cast<int>().cwiseMax(Vec3i(0,0,0));

                                                                      // ceil
    const Vec3i boundingBoxMax = (vertexInGridCoords + nSigmaExtent + Scalar(0.99999)*Vec3::Ones()).template cast<int>()
            .cwiseMin(nearestNeighborGrid.dimensions().template cast<int>());

    std::cout << nSigmaExtent.transpose() << std::endl;
    std::cout << vertex.transpose() << "  -->   " << vertexInGridCoords.transpose() << std::endl;
    std::cout << boundingBoxMin.transpose() << " -> " << boundingBoxMax.transpose() << std::endl;

    const Vec3i boundingBoxSize = boundingBoxMax - boundingBoxMin;

    const dim3 block(16,16,4);
    const dim3 grid(intDivideAndCeil(boundingBoxSize(0),(int)block.x),
                    intDivideAndCeil(boundingBoxSize(1),(int)block.y),
                    intDivideAndCeil(boundingBoxSize(2),(int)block.z));

    std::cout << grid.x << ", " << grid.y << ", " << grid.z << std::endl;

    updateDeformationGraphNearestNeighborsKernel<<<grid,block>>>(nearestNeighborGrid,deformationGraphVertices,boundingBoxMin,boundingBoxMax,vertexInGridCoords,index);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}


template <typename Scalar, template <typename,int...> class TransformT>
template <int K>
void NonrigidTransformer<Scalar,TransformT>::computeDeformationGraphNearestNeighbors(const Scalar nearestNeighborSigma) {

    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    NNVec initialNearestNeighborList = -1 * NNVec::Ones();

    nearestNeighborGrid_->fill(initialNearestNeighborList);

//    const uint numBaseLevelVertices = numVerticesAtLevel(0);

//    ManagedDeviceTensor1<Vec3> baseLevelVertices( numBaseLevelVertices );

//    ConstHostTensor1<Vec3> hostBaseLevelVertices(baseLevelVertices.length(), deformationGraphVertices(0) );

//    baseLevelVertices.copyFrom(hostBaseLevelVertices);

    updateDeviceVertices();

    for (uint index = 0; index < numVerticesAtLevel(0); ++index) {

        const Vec3 & vertex = deformationGraphVertices(0)[index];

        updateDeformationGraphNearestNeighbors(*nearestNeighborGrid_,deviceBaseLevelVertices_,vertex,index,nearestNeighborSigma);

    }

}

template void NonrigidTransformer<float,DualQuaternion>::computeDeformationGraphNearestNeighbors<4>(const float nearestNeighborSigma);

template void NonrigidTransformer<float,Sophus::SE3>::computeDeformationGraphNearestNeighbors<4>(const float nearestNeighborSigma);




template <typename Scalar, int K>
__global__ void markUnsupportedVerticesKernel(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > vertices,
                                              const DeviceVoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign> > nearestNeighborGrid,
                                              const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > deformationGraphVertices,
                                              const Scalar maximumSupportDistanceSquared,
                                              DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > unsupportedVertices,
                                              int * __restrict__ nUnsupported) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<int,3,1,Eigen::DontAlign> Vec3i;
    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;

    const uint i = threadIdx.x + blockDim.x * blockIdx.x;

    if ( i < vertices.length() ) {

        const Vec3 & vertexInGridCoords = vertices(i);

        const Vec3i discreteVoxel = round(vertexInGridCoords);

        if (!nearestNeighborGrid.grid().inBounds(discreteVoxel,0)) {

            return;

        }

        const NNVec & nearestNeighborIndices = nearestNeighborGrid(discreteVoxel);

        const Vec3 vertexInWorldCoords = nearestNeighborGrid.gridToWorld(vertexInGridCoords);

        // find first vertex
        if (nearestNeighborIndices(0) >= 0) {

            // there is at least one nearest neighbor
            // k is the closest; how far is it?
            const int neighborIndex = nearestNeighborIndices(0);

            const Vec3 & neighborVertex = deformationGraphVertices(neighborIndex);

            const Scalar distanceSquared = (vertexInWorldCoords - neighborVertex).squaredNorm();

//            printf("%d: %f vs %f\n",i,distanceSquared,maximumSupportDistanceSquared);

//            printf("%d %d\n",i,ne)

            if (distanceSquared <= maximumSupportDistanceSquared) {

                // we're covered
                return;

            }

        }

        const int index = atomicAdd(nUnsupported, 1);

        if (index < unsupportedVertices.length()) {

            unsupportedVertices(index) = vertexInWorldCoords;

        }

    }

}

template <typename Scalar, int K>
__global__ void initializeNewBaseLevelVertexTransformsKernel(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > baseLevelVertices,
                                                             const DeviceVoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign> > nearestNeighborGrid,
                                                             DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > baseLevelTransforms,
                                                             const int numExistingBaseLevelVertices,
                                                             const Scalar oneOverBlendingSigmaSquared) {

    typedef Eigen::UnalignedVec3<Scalar> Vec3;

    const uint threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

    const uint index = numExistingBaseLevelVertices + threadIndex;

    if (index < baseLevelVertices.length()) {

        printf("updating transform %d\n",index);

        const Vec3 & newVertexInWorldCoords = baseLevelVertices(index);

        const Vec3 newVertexInGridCoords = nearestNeighborGrid.worldToGrid(newVertexInWorldCoords);


        const bool initialized = blendDualQuaternions(baseLevelTransforms(index),
                                                      newVertexInGridCoords,newVertexInWorldCoords,
                                                      baseLevelVertices, baseLevelTransforms, nearestNeighborGrid,
                                                      oneOverBlendingSigmaSquared);

        if (initialized) {

            printf("initialized %d\n",index);

            baseLevelTransforms(index).normalize();

        }

    }

}

template <typename Scalar, template <typename, int...> class TransformT>
void NonrigidTransformer<Scalar,TransformT>::update(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
                                                    const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & unsupportedVertices,
                                                    int & hNumUnsupportedVertices,
                                                    const int minUnsupportedVertices) {

//    ManagedDeviceTensor1<int> unsupportedIndices( vertices.length() * 0.1 ); // TODO
    ManagedDeviceTensor1<int> numUnsupportedVertices( 1 );
    cudaMemset(numUnsupportedVertices.data(), 0, sizeof(int));

    const dim3 block(1024);
    const dim3 grid(intDivideAndCeil(vertices.length(), block.x));

    updateDeviceVertices();

    const Scalar maximumSupportDistance = 1.2*baseDecimationResolution_;

    markUnsupportedVerticesKernel<<<grid,block>>>(vertices, *nearestNeighborGrid_, deviceBaseLevelVertices_, maximumSupportDistance*maximumSupportDistance, unsupportedVertices, numUnsupportedVertices.data());

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

//    int hNumUnsupportedVertices;
    cudaMemcpy(&hNumUnsupportedVertices, numUnsupportedVertices.data(), sizeof(int), cudaMemcpyDeviceToHost);

    hNumUnsupportedVertices = std::min( hNumUnsupportedVertices, (int) unsupportedVertices.length() );

    std::cout << hNumUnsupportedVertices << " unsupported" << std::endl;

    if (hNumUnsupportedVertices >= minUnsupportedVertices) {

        ManagedHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > hUnsupportedVertices(hNumUnsupportedVertices);

        hUnsupportedVertices.copyFrom( ConstDeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> >( hNumUnsupportedVertices, unsupportedVertices.data() ) );

        const uint numExistingBaseLevelVertices = deformationGraphVertices_[0].size();

        ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > existingBaseLevelVertices( numExistingBaseLevelVertices, deformationGraphVertices_[0].data() );

        deformationGraphVertices_[0].resize(numExistingBaseLevelVertices + hNumUnsupportedVertices);

        HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > newDecimatedVertices( hNumUnsupportedVertices, deformationGraphVertices_[0].data() + numExistingBaseLevelVertices );

        const uint numNewBaseLevelVertices = decimateIncremental<Scalar>( existingBaseLevelVertices,
                                                                          hUnsupportedVertices,
                                                                          newDecimatedVertices,
                                                                          baseDecimationResolution_ );

        std::cout << numNewBaseLevelVertices << " new level 0 vertices" << std::endl;

        const uint numBaseLevelVertices = numExistingBaseLevelVertices + numNewBaseLevelVertices;

        deformationGraphVertices_[0].resize( numBaseLevelVertices );
        vertexTransforms_[0].resize( numBaseLevelVertices );

        if (numNewBaseLevelVertices > 0) {

            deviceVerticesCurrent_ = false;
            deviceTransformsCurrent_ = false;

            updateDeviceVerticesAndTransforms();

            // initialize base level transforms using current nearest neighbors
            const dim3 block(32);
            const dim3 grid(intDivideAndCeil(numNewBaseLevelVertices,block.x));

            initializeNewBaseLevelVertexTransformsKernel<<<grid,block>>>(deviceBaseLevelVertices_,
                                                                         *nearestNeighborGrid_,
                                                                         deviceBaseLevelDualQuaternionTransforms_,
                                                                         numExistingBaseLevelVertices,
                                                                         Scalar(1) / (baseDecimationResolution_*baseDecimationResolution_));

            // copy new base level transforms back to host
            ManagedHostTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > newBaseLevelTransforms( numNewBaseLevelVertices );
            newBaseLevelTransforms.copyFrom( DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> >( numNewBaseLevelVertices, deviceBaseLevelDualQuaternionTransforms_.data() + numExistingBaseLevelVertices) );

            for (uint index = numExistingBaseLevelVertices; index < numBaseLevelVertices; ++index) {

                const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertex = deformationGraphVertices_[0][index];
                const DualQuaternion<Scalar,Eigen::DontAlign> decenterer(Eigen::Quaternion<Scalar,Eigen::DontAlign>(1,0,0,0),
                                                                         Eigen::Quaternion<Scalar,Eigen::DontAlign>(0,vertex(0)/2,vertex(1)/2,vertex(2)/2));
                const DualQuaternion<Scalar,Eigen::DontAlign> centerer(Eigen::Quaternion<Scalar,Eigen::DontAlign>(1,0,0,0),
                                                                       Scalar(-1)*Eigen::Quaternion<Scalar,Eigen::DontAlign>(0,vertex(0)/2,vertex(1)/2,vertex(2)/2));

                vertexTransforms_[0][index] = centerer*newBaseLevelTransforms(index - numExistingBaseLevelVertices)*decenterer;

            }

            // update nearest neighbors
            for (uint index = numExistingBaseLevelVertices; index < numBaseLevelVertices; ++index ) {

                const Vec3 & vertex = deformationGraphVertices_[0][index];

                updateDeformationGraphNearestNeighbors(*nearestNeighborGrid_,deviceBaseLevelVertices_,
                                                       vertex, index, 0.5f*baseDecimationResolution_);

            }

            uint numExistingPreviousLevelVertices = numExistingBaseLevelVertices;

            uint numNewPreviousLevelVertices = numNewBaseLevelVertices;

            for (uint level = 1; level < numRegularizationTreeLevels(); ++level) {

                const uint numExistingThisLevelVertices = deformationGraphVertices_[level].size();

                deformationGraphVertices_[level].resize(numExistingThisLevelVertices + numNewPreviousLevelVertices);

                const Scalar thisLevelRadius = baseDecimationResolution_ * pow(levelToLevelScale_, level);

                HostTensor1<Vec3> existingThisLevelVertices( numExistingThisLevelVertices, deformationGraphVertices_[level].data() );

                HostTensor1<Vec3> newPreviousLevelVertices( numNewPreviousLevelVertices, deformationGraphVertices_[level-1].data() + numExistingPreviousLevelVertices );

                HostTensor1<Vec3> newThisLevelVertices( numNewPreviousLevelVertices, deformationGraphVertices_[level].data() + numExistingThisLevelVertices);

                const uint numNewThisLevelVertices = decimateIncremental<Scalar>( existingThisLevelVertices,
                                                                                  newPreviousLevelVertices,
                                                                                  newThisLevelVertices,
                                                                                  thisLevelRadius);

                std::cout << numNewThisLevelVertices << " new level " << level << " vertices" << std::endl;

                const uint numThisLevelVertices = numExistingThisLevelVertices + numNewThisLevelVertices;

                deformationGraphVertices_[level].resize( numThisLevelVertices );

                // compute neighbors
                std::vector<std::vector<uint> > & thisLevelLowerLevelNeighbors = lowerLevelNeighbors_[level-1]; // -1 because the base layer doesn't have lower level neighbors
                std::vector<std::vector<uint> > & previousLevelHigherLevelNeighbors = higherLevelNeighbors_[level-1];

                thisLevelLowerLevelNeighbors.resize( numThisLevelVertices );

                const uint numPreviousLevelVertices = deformationGraphVertices_[level-1].size();

                previousLevelHigherLevelNeighbors.resize( numPreviousLevelVertices );

                HostTensor1<Vec3> allThisLevelVertices( deformationGraphVertices_[level].size(), deformationGraphVertices_[level].data() );

                KDPointCloud<Scalar> allThisLevelVerticesCloud( allThisLevelVertices );

                KDTree<Scalar> allThisLevelVerticesTree(3, allThisLevelVerticesCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

                allThisLevelVerticesTree.buildIndex();

                std::vector<int> nearestNeighborIndices(numRegularizationNeighbors_);
                std::vector<Scalar> nearestNeighborDistancesSquared(numRegularizationNeighbors_);

                // connect new previous level vertices to ALL this layer vertices
                for (uint previousLevelIndex = numExistingPreviousLevelVertices; previousLevelIndex < numPreviousLevelVertices; ++previousLevelIndex) {

                    const Vec3 & previousLevelVertex = deformationGraphVertices_[level-1][previousLevelIndex];

                    allThisLevelVerticesTree.knnSearch(previousLevelVertex.data(), numRegularizationNeighbors_,
                                                       nearestNeighborIndices.data(), nearestNeighborDistancesSquared.data());

                    for (uint k = 0; k < numRegularizationNeighbors_; ++k) {

                        const uint thisLevelIndex = nearestNeighborIndices[k];

                        thisLevelLowerLevelNeighbors[thisLevelIndex].push_back(previousLevelIndex);

                        previousLevelHigherLevelNeighbors[previousLevelIndex].push_back(thisLevelIndex);

                    }

                }

                // check to see if any previous level vertices need to be connected to this level vertices
                for (uint previousLevelIndex = 0; previousLevelIndex < numExistingPreviousLevelVertices; ++previousLevelIndex) {

                    const Vec3 & previousLevelVertex = deformationGraphVertices_[level-1][previousLevelIndex];

                    std::vector<uint> & previousLevelVertexNeighbors = previousLevelHigherLevelNeighbors[previousLevelIndex];

                    std::vector<Scalar> previousLevelNeighborDistancesSquared(previousLevelVertexNeighbors.size());

                    //  compute distances to current neighbors
                    for (int k = 0; k < previousLevelVertexNeighbors.size(); ++k) {

                        const uint thisLevelIndex = previousLevelVertexNeighbors[k];

                        const Vec3 & thisLevelVertex = deformationGraphVertices_[level][thisLevelIndex];

                        previousLevelNeighborDistancesSquared[previousLevelVertexNeighbors.size()] = (thisLevelVertex - previousLevelVertex).squaredNorm();

                    }

                    // compute distance to all new vertices and update (sorted) neighbor list as needed
                    for (uint thisLevelIndex = numExistingThisLevelVertices; thisLevelIndex < numThisLevelVertices; ++thisLevelIndex) {

                        const Vec3 & thisLevelVertex = deformationGraphVertices_[level][thisLevelIndex];

                        const Scalar distanceSquared = (thisLevelVertex - previousLevelVertex).squaredNorm();

                        int insertionIndex;
                        for (insertionIndex = previousLevelVertexNeighbors.size(); insertionIndex >= 1 && previousLevelNeighborDistancesSquared[insertionIndex-1] > distanceSquared; --insertionIndex) { }

                        for (int k = previousLevelVertexNeighbors.size()-1; k > insertionIndex; --k) {
                            previousLevelVertexNeighbors[k] = previousLevelVertexNeighbors[k-1];
                            previousLevelNeighborDistancesSquared[k] = previousLevelNeighborDistancesSquared[k-1];
                        }

                        if (insertionIndex < previousLevelVertexNeighbors.size()) {
                            previousLevelVertexNeighbors[insertionIndex] = thisLevelIndex;
                            previousLevelNeighborDistancesSquared[insertionIndex] = distanceSquared;
                        }

                    }

                }

                // initialize transforms for this level's vertices
                vertexTransforms_[level].resize(numThisLevelVertices);
                for (uint thisLevelIndex = numExistingThisLevelVertices; thisLevelIndex < numThisLevelVertices; ++thisLevelIndex) {

                    // TODO: duplicated code from nonrigidDeviceModule.h

//                    const Vec3 & thisLevelVertex = deformationGraphVertices_[level][thisLevelIndex];

                    const std::vector<uint> & neighbors = lowerLevelNeighbors_[level-1][thisLevelIndex];

//                    Eigen::Quaternion<Scalar,Eigen::DontAlign> firstNondual;

//                    DualQuaternion<Scalar,Eigen::DontAlign> blendedTransform;

//                    for (int k = 0; k < neighbors.size(); ++k) {

//                        const uint lowerLevelIndex = neighbors[k];

//                        const Vec3 & lowerLevelVertex = deformationGraphVertices_[level-1][lowerLevelIndex];

//                        const Scalar distanceSquared = (lowerLevelVertex - thisLevelVertex).squaredNorm();

//                        const Scalar weight = expX(-distanceSquared / (baseDecimationResolution_ * baseDecimationResolution_) );

//                        const DualQuaternion<Scalar,Eigen::DontAlign> lowerLevelTransform = vertexTransforms_[level-1][lowerLevelIndex];

//                        if ( k == 0 ) {

//                            firstNondual = lowerLevelTransform.nondual();

//                            blendedTransform = weight * lowerLevelTransform;

//                        } else {

//                            if (lowerLevelTransform.nondual().dot(firstNondual) < Scalar(0)) {

//                                blendedTransform -= weight * lowerLevelTransform;

//                            } else {

//                                blendedTransform += weight * lowerLevelTransform;

//                            }

//                        }

//                    }

//                    blendedTransform.normalize();

//                    vertexTransforms_[level][thisLevelIndex] = blendedTransform;

                    vertexTransforms_[level][thisLevelIndex] = vertexTransforms_[level-1][neighbors[0]];

                    std::cout << "initialized transform " << thisLevelIndex << " to " << std::endl << vertexTransforms_[level][thisLevelIndex] << std::endl << std::endl;

                }

                numNewPreviousLevelVertices = numNewThisLevelVertices;
                numExistingPreviousLevelVertices = numExistingThisLevelVertices;

            }

            ++serialNumber_;

        }

    }

//    higherLevelNeighbors_[1].resize(deformationGraphVertices_[1].size());

    if (!invariantsHold()) {
        throw std::runtime_error("invariants do not hold");
    }

}

template class NonrigidTransformer<float,Sophus::SE3>;

template class NonrigidTransformer<float,DualQuaternion>;


} // namespace df
