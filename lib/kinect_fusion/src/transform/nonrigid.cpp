#include <df/transform/nonrigid.h>

#include <df/surface/decimation.h>
#include <df/util/nanoflannHelpers.h>
#include <df/util/dualQuaternion.h>
#include <df/util/sophusHelpers.h> // TODO

#include <nanoflann.hpp>
#include <memory>

namespace df {

// the "int..." bit is to allow use of Sophus::SE3Group, which has an extra template
// parameter of type int after the Scalar template
template <typename Scalar, template <typename,int...> class TransformT>
bool NonrigidTransformer<Scalar,TransformT>::invariantsHold() const {

    // deformationGraphNeighbors_ stores neighbor indices for the previous regularization
    // tree level. thus there should be one less vector, because the base level nodes
    // can have no neighbors
    if (higherLevelNeighbors_.size() != (deformationGraphVertices_.size()-1)) {
        std::cerr << "wrong number of higher level neighbors" << std::endl;
        return false;
    }
    if (lowerLevelNeighbors_.size() != (deformationGraphVertices_.size()-1)) {
        std::cerr << "wrong number of lower level neighbors" << std::endl;
        return false;
    }

    for (uint level = 0; level < deformationGraphVertices_.size()-1; ++level) {
        if (lowerLevelNeighbors_[level].size() != deformationGraphVertices_[level+1].size()) {
            return false;
        }
        if (higherLevelNeighbors_[level].size() != deformationGraphVertices_[level].size()) {
            return false;
        }

        // check reciprocity of neighbors
        for (uint lowerLevelIndex = 0; lowerLevelIndex < deformationGraphVertices_[level].size(); ++lowerLevelIndex) {

            for (uint k = 0; k < higherLevelNeighbors_[level][lowerLevelIndex].size(); ++k) {

                const uint higherLevelIndex = higherLevelNeighbors_[level][lowerLevelIndex][k];

                const std::vector<uint> & neighbors = lowerLevelNeighbors_[level][higherLevelIndex];

                if (std::find(neighbors.begin(),neighbors.end(),lowerLevelIndex) == neighbors.end()) {
                    return false;
                }

            }

        }
    }

    if (vertexTransforms_.size() != deformationGraphVertices_.size()) {
        return false;
    }

    for (uint level = 0; level < deformationGraphVertices_.size(); ++level) {

        std::cout << "checking level " << level << std::endl;

        if (deformationGraphVertices_[level].size() != vertexTransforms_[level].size()) {
            return false;
        }

        std::cout << vertexTransforms_[level].size() << std::endl;

    }

    return true;
}

template <typename Scalar, template <typename,int...> class TransformT>
void NonrigidTransformer<Scalar,TransformT>::initialize(const HostTensor1<Vec3> & vertices,
                                                        const Scalar baseDecimationResolution,
                                                        const uint numRegularizationTreeLevels,
                                                        const Scalar levelToLevelScale,
                                                        const Eigen::Matrix<uint,3,1> & gridDimensions,
                                                        const Eigen::AlignedBox<Scalar,3> & gridBoundingBox,
                                                        const uint numRegularizationNeighbors) {

    using namespace operators;

//    deformationGraphVertices_.resize(2);
//    lowerLevelNeighbors_.resize(1);
//    higherLevelNeighbors_.resize(1);

//    deformationGraphVertices_[0].resize(4);
//    deformationGraphVertices_[1].resize(1);

//    deformationGraphVertices_[1][0] = Vec3(0,0,0);

//    deformationGraphVertices_[0][0] = Vec3(0,0,10);
//    deformationGraphVertices_[0][1] = Vec3(10,-10,-10);
//    deformationGraphVertices_[0][2] = Vec3(-10,-10,-10);
//    deformationGraphVertices_[0][3] = Vec3(0,10,-10);

//    higherLevelNeighbors_[0] = { {0}, {0}, {0}, {0} };
//    lowerLevelNeighbors_[0] = { {0,1,2,3} };

    assert(numRegularizationTreeLevels > 0);

    deformationGraphVertices_.resize(numRegularizationTreeLevels);
    lowerLevelNeighbors_.resize(numRegularizationTreeLevels-1);
    higherLevelNeighbors_.resize(numRegularizationTreeLevels-1);

    // compute the base-level deformation graph vertices
    deformationGraphVertices_[0].resize(vertices.length());

    HostTensor1<Vec3> baseLevelVertices(vertices.length(), deformationGraphVertices_[0].data());

    const uint numBaseLevelVertices = decimate(vertices,baseLevelVertices,baseDecimationResolution);

    deformationGraphVertices_[0].resize(numBaseLevelVertices);

    HostTensor1<Vec3> baseLevelVerticesPacked( deformationGraphVertices_[0].size(), deformationGraphVertices_[0].data() );
    std::unique_ptr<KDPointCloud<Scalar> > previousLevelPointCloud(new KDPointCloud<Scalar>(baseLevelVerticesPacked));
    std::unique_ptr<KDTree<Scalar> > previousLevelKDTree(new KDTree<Scalar>(3, *previousLevelPointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
    previousLevelKDTree->buildIndex();

    // build higher levels of deformation graph
    for (uint level = 1; level < numRegularizationTreeLevels; ++level) {

        // make room for new vertices, with absolute worst case being 1 vertex for each vertex in the previous level
        deformationGraphVertices_[level].resize(deformationGraphVertices_[level-1].size());

        HostTensor1<Vec3> previousLevelVertices( deformationGraphVertices_[level-1].size(), deformationGraphVertices_[level-1].data() );

        HostTensor1<Vec3> thisLevelVertices( deformationGraphVertices_[level].size(), deformationGraphVertices_[level].data() );

        const Scalar thisLevelRadius = baseDecimationResolution*pow(levelToLevelScale,level);

        const uint numThisLevelVertices = decimate(previousLevelVertices,*previousLevelKDTree,thisLevelVertices,thisLevelRadius);

        // re-size the current level to match the actual number of vertices
        deformationGraphVertices_[level].resize(numThisLevelVertices);

        HostTensor1<Vec3> thisLevelVerticesPacked( deformationGraphVertices_[level].size(), deformationGraphVertices_[level].data() );
        std::unique_ptr<KDPointCloud<Scalar> > thisLevelPointCloud(new KDPointCloud<Scalar>(thisLevelVerticesPacked));
        std::unique_ptr<KDTree<Scalar> > thisLevelKDtree(new KDTree<Scalar>(3, *thisLevelPointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
        thisLevelKDtree->buildIndex();

        // compute neighbors
        const uint numPreviousLevelVertices = deformationGraphVertices_[level-1].size();

        std::vector<std::vector<uint> > & thisLevelLowerLevelNeighbors = lowerLevelNeighbors_[level-1]; // -1 because the base layer doesn't have lower level neighbors
        std::vector<std::vector<uint> > & previousLevelHigherLevelNeighbors = higherLevelNeighbors_[level-1];

        thisLevelLowerLevelNeighbors.resize(numThisLevelVertices);
        previousLevelHigherLevelNeighbors.resize(numPreviousLevelVertices);

        std::vector<int> nearestNeighborIndices(numRegularizationNeighbors);
        std::vector<Scalar> nearestNeighborDistancesSquared(numRegularizationNeighbors);

        for (uint previousLevelIndex = 0; previousLevelIndex < numPreviousLevelVertices; ++previousLevelIndex) {

            const Vec3 & previousLevelVertex = deformationGraphVertices_[level-1][previousLevelIndex];

            thisLevelKDtree->knnSearch(previousLevelVertex.data(),numRegularizationNeighbors,
                                       nearestNeighborIndices.data(),nearestNeighborDistancesSquared.data());

            for (uint k = 0; k < numRegularizationNeighbors; ++k) {

                const uint thisLevelIndex = nearestNeighborIndices[k];

                thisLevelLowerLevelNeighbors[thisLevelIndex].push_back(previousLevelIndex);

                previousLevelHigherLevelNeighbors[previousLevelIndex].push_back(thisLevelIndex);

            }

        }

        previousLevelPointCloud.swap(thisLevelPointCloud);
        previousLevelKDTree.swap(thisLevelKDtree);

    }

    // initalize transforms
    vertexTransforms_.resize(deformationGraphVertices_.size());
    for (uint i = 0; i < deformationGraphVertices_.size(); ++i) {
        vertexTransforms_[i].resize(deformationGraphVertices_[i].size());
        for (uint j = 0; j < deformationGraphVertices_[i].size(); ++j) {
            vertexTransforms_[i][j] = Transform();
            std::cout << vertexTransforms_[i][j] << std::endl;
            std::cout << sizeof(Transform) << std::endl;
            std::cout << ((int64_t)&vertexTransforms_[0][1] - (int64_t)&vertexTransforms_[0][0]) << std::endl;
            std::cout << std::alignment_of<Transform>::value << std::endl;
        }
    }


    for (uint i = 0; i < deformationGraphVertices_.size(); ++i) {
        std::cout << deformationGraphVertices_[i].size() << " level " << i << " vertices" << std::endl;
    }

    // initialize sigma

    ++serialNumber_;

    delete nearestNeighborData_;
    nearestNeighborData_ = new ManagedDeviceTensor3<Eigen::UnalignedVec<int,4> >(gridDimensions);

    delete nearestNeighborGrid_;
    nearestNeighborGrid_ = new VoxelGrid<Scalar,Eigen::UnalignedVec<int,4>,DeviceResident>(
                gridDimensions, nearestNeighborData_->data(), gridBoundingBox);

    // TODO: decouple NN sigma from decimation resolution?
    computeDeformationGraphNearestNeighbors<4>(0.5f * baseDecimationResolution);

    baseDecimationResolution_ = baseDecimationResolution;
    levelToLevelScale_ = levelToLevelScale;
    numRegularizationNeighbors_ = numRegularizationNeighbors;

    deviceVerticesCurrent_ = false;
    deviceTransformsCurrent_ = false;

//    assert(invariantsHold());
    if (!invariantsHold()) {
        throw std::runtime_error("invariants violated");
    }

}

#define NONRIGID_DQ_TRANSFORMER_EXPLICIT_INSTANTIATION(type)     \
    template class NonrigidTransformer<type,DualQuaternion>

#define NONRIGID_SE3_TRANSFORMER_EXPLICIT_INSTANTIATION(type)     \
    template class NonrigidTransformer<type,Sophus::SE3>

ALL_TYPES_INSTANTIATION(NONRIGID_DQ_TRANSFORMER_EXPLICIT_INSTANTIATION);
ALL_TYPES_INSTANTIATION(NONRIGID_SE3_TRANSFORMER_EXPLICIT_INSTANTIATION);

} // namespace df
