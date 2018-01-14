#pragma once

#include <df/transform/rigid.h>
#include <df/util/dualQuaternion.h>
#include <df/util/eigenHelpers.h>
#include <df/util/macros.h>
#include <df/util/tensor.h>
#include <df/voxel/voxelGrid.h>

#include <iostream>
#include <vector>

#include <thrust/host_vector.h>

#include <Eigen/Core>

namespace df {

template <typename Scalar, template <typename,int...> class TransformT>
class NonrigidTransformer : public RigidTransformer<Scalar> {
public:

    typedef typename RigidTransformer<Scalar>::Vec3 Vec3;
    typedef TransformT<Scalar> Transform;

    NonrigidTransformer()
        : nearestNeighborData_(nullptr),
          nearestNeighborGrid_(nullptr),
          serialNumber_(0),
          deviceVerticesCurrent_(false),
          deviceTransformsCurrent_(false) { }

    ~NonrigidTransformer() {
        delete nearestNeighborData_;
        delete nearestNeighborGrid_;
    }

    void initialize(const HostTensor1<Vec3> & vertices,
                    const Scalar baseDecimationResolution,
                    const uint numRegularizationTreeLevels,
                    const Scalar levelToLevelScale,
                    const Eigen::Matrix<uint,3,1> & gridDimension,
                    const Eigen::AlignedBox<Scalar,3> & gridBoundingBox,
                    const uint numRegularizationNeighbors = 4);

    void update(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
                const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & unsupportedIndices,
                int & numUnsupportedIndices,
                const int minUnsupportedVertices = 10);

    inline uint numRegularizationTreeLevels() const {
        return deformationGraphVertices_.size();
    }

    inline uint numVerticesAtLevel(const uint level) const {
        assert(level < deformationGraphVertices_.size());
        return deformationGraphVertices_[level].size();
    }

    inline const Vec3 * deformationGraphVertices(const uint level) const {
        assert(level < deformationGraphVertices_.size());
        return deformationGraphVertices_[level].data();
    }

    inline uint numHigherLevelNeighbors(const uint level, const uint lowerLevelIndex) const {
        assert(level < higherLevelNeighbors_.size());
        assert(lowerLevelIndex < higherLevelNeighbors_[level].size());
        return higherLevelNeighbors_[level][lowerLevelIndex].size();
    }

    inline const uint * higherLevelNeighbors(const uint level, const uint lowerLevelIndex) const {
        assert(level < higherLevelNeighbors_.size());
        assert(lowerLevelIndex < higherLevelNeighbors_[level].size());
        return higherLevelNeighbors_[level][lowerLevelIndex].data();
    }

    inline uint numLowerLevelNeighbors(const uint level, const uint higherLevelIndex) const {
        assert(level > 0 && level < deformationGraphVertices_.size());
        assert(higherLevelIndex < lowerLevelNeighbors_[level-1].size());
        return lowerLevelNeighbors_[level-1][higherLevelIndex].size();
    }

    inline const uint * lowerLevelNeighbors(const uint level, const uint higherLevelIndex) const {
        assert(level > 0 && level < deformationGraphVertices_.size());
        assert(higherLevelIndex < lowerLevelNeighbors_[level-1].size());
        return lowerLevelNeighbors_[level-1][higherLevelIndex].data();
    }

    inline Transform * transforms(const uint level) {
        deviceTransformsCurrent_ = false; // TODO: right place for this?
        assert( level < deformationGraphVertices_.size());
        return vertexTransforms_[level].data();
    }

    inline const Transform * transforms(const uint level) const {
        assert( level < deformationGraphVertices_.size());
        return vertexTransforms_[level].data();
    }

    inline uint numVerticesTotal() const {
        return std::accumulate(deformationGraphVertices_.begin(),deformationGraphVertices_.end(),
                               0,[](const uint & runningTotal, const std::vector<Vec3> & levelVertices){
                                    return runningTotal + levelVertices.size(); });
    }

    template <int K, int DTensor, int DVec>
    void warpMesh(DeviceTensor<DTensor,Eigen::UnalignedVec3<Scalar> > & warpedVertices,
                  DeviceTensor<DTensor,Eigen::UnalignedVec3<Scalar> > & warpedNormals,
                  const DeviceTensor<DTensor,Eigen::UnalignedVec<Scalar,DVec> > & unwarpedVertices,
                  const DeviceTensor<DTensor,Eigen::UnalignedVec<Scalar,DVec> > & unwarpedNormals);

    // this number increments every time the deformation graph is changed.
    // this allows, for example, the solver to know when to re-do the
    // symbolic analysis and when it can use the same analysis from the
    // last iteration.
    inline uint serialNumber() const {
        return serialNumber_;
    }

    inline Scalar blendingSigma() const {

        return baseDecimationResolution_;

    }

    bool invariantsHold() const;

    inline const VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident> & nearestNeighborGrid() const {
        return *nearestNeighborGrid_;
    }

    inline DeviceTensor1<Vec3> deviceBaseLevelVertices() {

        updateDeviceVertices();

        return deviceBaseLevelVertices_;

    }

    inline DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > deviceBaseLevelDualQuaternionTransforms() {

        updateDeviceTransforms();

        return deviceBaseLevelDualQuaternionTransforms_;

    }

    class DeviceModule {
    public:

        inline DeviceModule(const Sophus::SE3<Scalar> & transformWorldToLive,
                            const VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident> & nearestNeighborGrid,
                            const DeviceTensor1<Vec3> & deviceBaseLevelVertices,
                            const DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & deviceBaseLevelTransforms,
                            const Scalar blendingSigma);

        inline __host__ __device__ Vec3 transformWorldToLive(const Vec3 & vertexInWorldCoords) const;

    private:

        const Sophus::SE3<Scalar> transformWorldToLive_;

        const VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident> nearestNeighborGrid_;

        const DeviceTensor1<Vec3> deviceBaseLevelVertices_;

        const DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > deviceBaseLevelTransforms_;

        const Scalar blendingSigma_;

    };

    inline DeviceModule deviceModule() const {

        updateDeviceVerticesAndTransforms();

        return DeviceModule(this->worldToLiveTransformation(),*nearestNeighborGrid_,
                            deviceBaseLevelVertices_, deviceBaseLevelDualQuaternionTransforms_,
                            baseDecimationResolution_);

    }

private:

    inline void updateDeviceVertices() const {

        if (!deviceVerticesCurrent_) {

            std::cout << "updating device verts" << std::endl;

            if (deviceBaseLevelVertices_.length() != numVerticesAtLevel(0)) {

                deviceBaseLevelVertices_.resize( numVerticesAtLevel(0) );

            }

            deviceBaseLevelVertices_.copyFrom(ConstHostTensor1<Vec3>( numVerticesAtLevel(0), deformationGraphVertices(0) ));


            deviceVerticesCurrent_ = true;
        }

    }

    inline void updateDeviceTransforms() const {

        if (!deviceTransformsCurrent_) {

            std::cout << "updating device transforms" << std::endl;

            if (deviceBaseLevelDualQuaternionTransforms_.length() != numVerticesAtLevel(0) ) {

                deviceBaseLevelDualQuaternionTransforms_.resize( numVerticesAtLevel(0) );

            }

            toDeviceRecenteredDualQuaternions(deviceBaseLevelDualQuaternionTransforms_,
                                              ConstHostTensor1<Vec3>( numVerticesAtLevel(0), deformationGraphVertices(0) ),
                                              ConstHostTensor1<Transform>( numVerticesAtLevel(0), transforms(0) ));

            deviceTransformsCurrent_ = true;
        }

    }

    inline void updateDeviceVerticesAndTransforms() const {

        updateDeviceVertices();

        updateDeviceTransforms();

    }

    template <int K>
    void computeDeformationGraphNearestNeighbors(const Scalar nearestNeighborSigma);

    void toDeviceRecenteredDualQuaternions(DeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > & dualQuaternions,
                                           const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & transformationOrigins,
                                           const ConstHostTensor1<TransformT<Scalar> > & transforms) const;

    std::vector<std::vector<Vec3> > deformationGraphVertices_;

    std::vector<std::vector<std::vector<uint> > > higherLevelNeighbors_;
    std::vector<std::vector<std::vector<uint> > > lowerLevelNeighbors_;


//    std::vector<thrust::host_vector<Transform,Eigen::aligned_allocator<Transform> > > vertexTransforms_;
    std::vector<EigenAlignedVector<Transform> > vertexTransforms_;

    ManagedDeviceTensor3<Eigen::UnalignedVec<int,4> > * nearestNeighborData_; // TODO

    VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident> * nearestNeighborGrid_;

    Scalar baseDecimationResolution_;
    Scalar levelToLevelScale_;
    int numRegularizationNeighbors_;

    uint serialNumber_;

    mutable ManagedDeviceTensor1<Vec3> deviceBaseLevelVertices_;

    mutable ManagedDeviceTensor1<DualQuaternion<Scalar,Eigen::DontAlign> > deviceBaseLevelDualQuaternionTransforms_;

    mutable bool deviceVerticesCurrent_, deviceTransformsCurrent_;
};


} // namespace df

#include <df/transform/nonrigidDeviceModule.h>
