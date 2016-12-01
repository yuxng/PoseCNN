#include <df/surface/decimation.h>

#include <df/util/macros.h>

#include <nanoflann.hpp>

namespace df {

template <typename Scalar>
uint decimate(const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
              HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & decimatedVertices,
              const Scalar radius,
              const int initialIndex) {

    KDPointCloud<Scalar> pointCloud(vertices);
    KDTree<Scalar> tree(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    return decimate(vertices,tree,decimatedVertices,radius,initialIndex);
}

template <typename Scalar>
uint decimate(const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
              KDTree<Scalar> & tree,
              HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & decimatedVertices,
              const Scalar radius,
              const int initialIndex) {

    const uint numVertices = vertices.length();

    nanoflann::SearchParams params;

    const Scalar radiusSquared = radius*radius;

    std::vector<std::pair<int,Scalar> > pointIndexDistancePairs;

    ManagedHostTensor1<bool> covered( numVertices - initialIndex );
    std::memset(covered.data(),0,covered.count()*sizeof(bool));

    uint nextDecimatedPoint = 0;

    for (uint i = initialIndex; i < numVertices; ++i) {

        if (!covered(i - initialIndex)) {

            const std::size_t numMatches = tree.radiusSearch(vertices(i).data(),radiusSquared,pointIndexDistancePairs,params);

            bool precovered = false;
            for (uint j = 0; j < numMatches; ++j) {
                if ( pointIndexDistancePairs[j].first >= initialIndex ) {
                    covered(pointIndexDistancePairs[j].first - initialIndex) = true;
                } else {
                    precovered = true;
                    break;
                }
            }

            if (!precovered) {
                decimatedVertices(nextDecimatedPoint) = vertices(i);
                ++nextDecimatedPoint;
            }

        }

    }

    return nextDecimatedPoint;
}

#define DECIMATE_EXPLICIT_INSTANTIATION(type)                                               \
    template uint decimate(const HostTensor1<Eigen::Matrix<type,3,1,Eigen::DontAlign> > &,  \
                           HostTensor1<Eigen::Matrix<type,3,1,Eigen::DontAlign> > &,        \
                           const type, const int)

ALL_TYPES_INSTANTIATION(DECIMATE_EXPLICIT_INSTANTIATION);



template <typename Scalar>
uint decimateIncremental(const ConstHostTensor1<Eigen::Matrix<Scalar,3,1, Eigen::DontAlign> > & originalVertices,
                         const ConstHostTensor1<Eigen::Matrix<Scalar,3,1, Eigen::DontAlign> > & newVertices,
                         HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & newDecimatedVertices,
                         const Scalar radius) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    // TODO
    ManagedHostTensor1<Vec3> allVertices( originalVertices.length() + newVertices.length() );

    HostTensor1<Vec3>( originalVertices.length(), allVertices.data() ).copyFrom(originalVertices);

    HostTensor1<Vec3>( newVertices.length(), allVertices.data() + originalVertices.length() ).copyFrom(newVertices);

    return decimate(allVertices, newDecimatedVertices, radius, originalVertices.length() );

}

#define DECIMATE_INCREMENTAL_EXPLICIT_INSTANTIATION(type)                                                   \
template uint decimateIncremental(const ConstHostTensor1<Eigen::Matrix<type,3,1,Eigen::DontAlign> > &,      \
                                  const ConstHostTensor1<Eigen::Matrix<type,3,1,Eigen::DontAlign> > &,      \
                                  HostTensor1<Eigen::Matrix<type,3,1,Eigen::DontAlign> > &,                 \
                                  const type radius)


ALL_TYPES_INSTANTIATION(DECIMATE_INCREMENTAL_EXPLICIT_INSTANTIATION);

} // namespace df
