#pragma once

#include <df/transform/nonrigid.h> // TODO: don't need this
#include <df/util/cudaHelpers.h>

namespace df {

template <typename Scalar, int K>
inline __host__ __device__
bool blendDualQuaternions(DualQuaternion<Scalar,Eigen::DontAlign> & blendedTransform,
                          const Eigen::UnalignedVec3<Scalar> & vertexInGridCoords,
                          const Eigen::UnalignedVec3<Scalar> & vertexInWorldCoords,
                          const DeviceTensor1<Eigen::UnalignedVec3<Scalar> > & deformationGraphVertices,
                          const DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & deformationGraphTransforms,
                          const DeviceVoxelGrid<Scalar,Eigen::UnalignedVec<int,K> > & nearestNeighborGrid,
                          const Scalar oneOverBlendingSigmaSquared) {

    using namespace operators;

    typedef Eigen::UnalignedVec3<Scalar> Vec3;
    typedef Eigen::UnalignedVec3<int> Vec3i;
    typedef Eigen::UnalignedVec<int,K> NNVec;
    typedef Eigen::Quaternion<Scalar,Eigen::DontAlign> Quaternion;
    typedef DualQuaternion<Scalar,Eigen::DontAlign> DualQuaternion;

    const Vec3i nearestNeighborVoxel = round(vertexInGridCoords);

    const NNVec nearestNeighborIndices = nearestNeighborGrid(nearestNeighborVoxel);

    const Quaternion * firstNondual;

//    Scalar totalWeight;

    bool initialized = false;

    // TODO: unroll
    for (int k = 0; k < K; ++k) {

        const int neighborIndex = nearestNeighborIndices(k);

        if (neighborIndex < 0) {
            break;
        }

        const Vec3 & deformationGraphVertex = deformationGraphVertices(neighborIndex);

        const DualQuaternion & deformationGraphTransform = deformationGraphTransforms(neighborIndex);

        const Vec3 offset = vertexInWorldCoords - deformationGraphVertex;

        const Scalar distanceSquared = offset.squaredNorm();

        const Scalar weight = expX(-distanceSquared*oneOverBlendingSigmaSquared);

        // TODO: do we need this?
        if (weight > Scalar(0)) {

            if (!initialized) {

                blendedTransform = weight*deformationGraphTransform;

//                totalWeight = weight;

                firstNondual = &deformationGraphTransform.nondual();

                initialized = true;

            } else {

//                totalWeight += weight;

//                        if (deformationGraphTransform.nondual().dot(blendedTransform.nondual()) < Scalar(0)) {
                if (deformationGraphTransform.nondual().dot(*firstNondual) < Scalar(0)) {

                    blendedTransform -= weight*deformationGraphTransform;

                } else {

                    blendedTransform += weight*deformationGraphTransform;

                }

            }

        }

    }

    // TODO: divide by total weight?

    blendedTransform.normalize();

    return initialized;

}

template <typename Scalar, template <typename,int...> class TransformT>
NonrigidTransformer<Scalar,TransformT>::DeviceModule::DeviceModule(const Sophus::SE3<Scalar> & transformWorldToLive,
                                                                   const VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident> & nearestNeighborGrid,
                                                                   const DeviceTensor1<Vec3> & deviceBaseLevelVertices,
                                                                   const DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & deviceBaseLevelTransforms,
                                                                   const Scalar blendingSigma)
    : transformWorldToLive_(transformWorldToLive),
      nearestNeighborGrid_(nearestNeighborGrid),
      deviceBaseLevelVertices_(deviceBaseLevelVertices),
      deviceBaseLevelTransforms_(deviceBaseLevelTransforms),
      blendingSigma_(blendingSigma) { }



template <typename Scalar, template <typename,int...> class TransformT>
inline __host__ __device__ typename NonrigidTransformer<Scalar,TransformT>::Vec3
NonrigidTransformer<Scalar,TransformT>::DeviceModule::transformWorldToLive(const Vec3 & vertexInWorldCoords) const {

    const Vec3 vertexInGridCoords = nearestNeighborGrid_.worldToGrid(vertexInWorldCoords);

    DualQuaternion<Scalar,Eigen::DontAlign> blendedTransform;

    const bool initialized = blendDualQuaternions(blendedTransform, vertexInGridCoords, vertexInWorldCoords,
                                                  deviceBaseLevelVertices_, deviceBaseLevelTransforms_,
                                                  nearestNeighborGrid_, Scalar(1)/(blendingSigma_*blendingSigma_));

    if (initialized) {

        const Vec3 warpedVertex = blendedTransform * vertexInWorldCoords;

        return transformWorldToLive_ * warpedVertex;

    } else {

        return Vec3(0,0,0);

    }

}

//class DeviceModule {
//public:

//    inline __host__ __device__ Vec3 transformWorldToLive(const Vec3 & vertexInWorldCoords) {

//        const Vec3 warpedVertex = vertexInWorldCoords; // TODO

//        return transformWorldToLive_ * warpedVertex;

//    }

//private:

//    Sophus::SE3<Scalar> transformWorldToLive_;

//};


} // namespace df
