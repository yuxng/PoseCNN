#include <df/optimization/deformationGraphRegularization.h>

#include <df/util/cudaHelpers.h>

#include <sophus/se3.hpp> // TODO
#include <df/util/dualQuaternion.h> // TODO

namespace df {

//template <typename Scalar, int K>
//struct InsertionIndexSearchUnroller {

//    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    __host__ __device__
//    static inline void search(int & insertionIndex,
//                              const NNVec & currentNeighbors,
//                              const Vec3 & voxelCenter,
//                              const Scalar distance,
//                              const DeviceTensor1<Vec3> & deformationGraphVertices) {

//        const int neighborIndex = currentNeighbors(K-1);

//        if (neighborIndex >= 0) {

//            const Vec3 & neighborVertex = deformationGraphVertices(neighborIndex);

//            const Vec3 neighborDiff = voxelCenter - neighborVertex;

//            const Scalar neighborDistance = neighborDiff.squaredNorm();

//            if (neighborDistance > distance) {

//                insertionIndex = K-1;

//            } else {

//                return;

//            }

//        } else {

//            insertionIndex = K-1;

//        }

//        InsertionIndexSearchUnroller<Scalar,K-1>::search(insertionIndex, currentNeighbors.template head<K-1>(),
//                                                         voxelCenter, distance, deformationGraphVertices);

//    }

//};

//template <typename Scalar>
//struct InsertionIndexSearchUnroller<Scalar, 0> {

//    typedef Eigen::Matrix<int,0,1,Eigen::DontAlign> NNVec;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    __host__ __device__
//    static inline void search(int & /*insertionIndex*/,
//                              const NNVec & /*currentNeighbors*/,
//                              const Vec3 & /*voxelCenter*/,
//                              const Scalar /*distance*/,
//                              const DeviceTensor1<Vec3> & /*deformationGraphVertices*/) { }

//};

//// TODO: make work for anisotropic grids
//template <typename Scalar, int K>
//__global__ void updateDeformationGraphNearestNeighborsKernel(Tensor<3,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> nearestNeighborGrid,
//                                                             const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > deformationGraphVertices,
//                                                             const Eigen::Matrix<int,3,1,Eigen::DontAlign> offset,
//                                                             const Eigen::Matrix<int,3,1,Eigen::DontAlign> max,
//                                                             const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> vertex, const int index) {

//    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    const int x = offset(0) + threadIdx.x + blockDim.x * blockIdx.x;
//    const int y = offset(1) + threadIdx.y + blockDim.y * blockIdx.y;
//    const int z = offset(2) + threadIdx.z + blockDim.z * blockIdx.z;

//    if (x < max(0) && y < max(1) && z < max(2)) {

////        printf("checking %d,%d,%d\n",x,y,z);

//        const Vec3 diff = Vec3(x,y,z) - vertex;

//        const Scalar distance = diff.squaredNorm();

////        printf("distance = %f\n",distance);

//        NNVec & currentNeighbors = nearestNeighborGrid(x,y,z);

////        printf("current = %d %d %d %d\n",currentNeighbors(0),currentNeighbors(1),currentNeighbors(2),currentNeighbors(3));

//        int insertionIndex = -1;

//        InsertionIndexSearchUnroller<Scalar,K>::search(insertionIndex,currentNeighbors,Vec3(x,y,z),
//                                                       distance, deformationGraphVertices);

////        for (int k = K-1; k >= 0; --k) {

////            const int neighborIndex = currentNeighbors(k);

////            if ( neighborIndex >= 0 ) {

////                const Eigen::Map<const Vec3> neighborVertex(&deformationGraphVertices(0,neighborIndex));

////                const Vec3 neighborDiff = Vec3(x,y,z) - neighborVertex;

////                const Scalar neighborDistance = neighborDiff.squaredNorm();

////                if (neighborDistance > distance) {

////                    // inserted index is closer
////                    insertionIndex = k;

////                } else {

////                    // inserted index is farther, the search ends
////                    break;

////                }


////            } else {

////                insertionIndex = k;

////            }

////        }

//        // check if the inserted vertex belongs in the updated nearest neighbor list
//        if (insertionIndex >= 0) {

//            for (int k = K-1; k > insertionIndex; --k) {

//                currentNeighbors(k) = currentNeighbors(k-1);

//            }

//            currentNeighbors(insertionIndex) = index;

//        }

//    }

//}

//template <typename Scalar, int K>
//void updateDeformationGraphNearestNeighbors(VoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> & nearestNeighborGrid,
//                                            const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & deformationGraphVertices,
//                                            const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & vertex, const int index,
//                                            const Scalar nearestNeighborSigma, const int nSigmas = 3) {

//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
//    typedef Eigen::Matrix<int,3,1,Eigen::DontAlign> Vec3i;

//    const Vec3 nSigmaExtent = nearestNeighborSigma * nSigmas * nearestNeighborGrid.worldToGridScale();

//    const Vec3 vertexInGridCoords = nearestNeighborGrid.worldToGrid(vertex);

//    const Vec3i boundingBoxMin = (vertexInGridCoords - nSigmaExtent).template cast<int>().cwiseMax(Vec3i(0,0,0));

//    const Vec3i boundingBoxMax = (vertexInGridCoords + nSigmaExtent + Scalar(0.99999)*Vec3::Ones()).template cast<int>()
//            .cwiseMin(nearestNeighborGrid.dimensions().template cast<int>() - Vec3i::Ones());

//    std::cout << vertex.transpose() << std::endl;
//    std::cout << boundingBoxMin.transpose() << " -> " << boundingBoxMax.transpose() << std::endl;

//    const Vec3i boundingBoxSize = boundingBoxMax - boundingBoxMin;

//    const dim3 block(16,16,4);
//    const dim3 grid(intDivideAndCeil(boundingBoxSize(0),(int)block.x),
//                    intDivideAndCeil(boundingBoxSize(1),(int)block.y),
//                    intDivideAndCeil(boundingBoxSize(2),(int)block.z));

//    updateDeformationGraphNearestNeighborsKernel<<<grid,block>>>(nearestNeighborGrid.grid(),deformationGraphVertices,boundingBoxMin,boundingBoxMax,vertex,index);

//    cudaDeviceSynchronize();
//    CheckCudaDieOnError();

//}

//template <typename Scalar, template <typename,int...> class TransformT, int K>
//void computeDeformationGraphNearestNeighbors(VoxelGrid<Scalar,Eigen::Matrix<int,K,1,Eigen::DontAlign>,DeviceResident> & nearestNeighborGrid,
//                                             const NonrigidTransformer<Scalar,TransformT> & transformer,
//                                             const Scalar nearestNeighborSigma) {

//    typedef Eigen::Matrix<int,K,1,Eigen::DontAlign> NNVec;
//    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

//    NNVec initialNearestNeighborList = -1 * NNVec::Ones();

//    nearestNeighborGrid.fill(initialNearestNeighborList);

//    const uint numBaseLevelVertices = transformer.numVerticesAtLevel(0);

//    ManagedDeviceTensor1<Vec3> baseLevelVertices( numBaseLevelVertices );

//    ConstHostTensor1<Vec3> hostBaseLevelVertices(baseLevelVertices.length(), transformer.deformationGraphVertices(0) );

//    baseLevelVertices.copyFrom(hostBaseLevelVertices);

//    for (uint index = 0; index < numBaseLevelVertices; ++index) {

//        const Vec3 & vertex = transformer.deformationGraphVertices(0)[index];

//        updateDeformationGraphNearestNeighbors(nearestNeighborGrid,baseLevelVertices,vertex,index,nearestNeighborSigma);

//    }

//}

//template void computeDeformationGraphNearestNeighbors(VoxelGrid<float,Eigen::Matrix<int,4,1,Eigen::DontAlign>,DeviceResident> &,
//                                                      const NonrigidTransformer<float,Sophus::SE3> &,
//                                                      const float);

//template void computeDeformationGraphNearestNeighbors(VoxelGrid<float,Eigen::Matrix<int,4,1,Eigen::DontAlign>,DeviceResident> &,
//                                                      const NonrigidTransformer<float,DualQuaternion> &,
//                                                      const float);

} // namespace df
