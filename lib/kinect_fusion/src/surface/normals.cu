#include <df/surface/normals.h>

#include <df/util/cudaHelpers.h>
#include <df/util/macros.h>

#include <df/voxel/color.h>
#include <df/voxel/probability.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/tsdf.h> // TODO

#include <Eigen/Geometry>

namespace df {

// TODO: maybe one thread per vertex, do a full-bandwidth read,
// compute normal with every third thread,
// broadcast, then do a full-bandwidth write?
template <typename Scalar>
__global__ void computeTriangularFaceNormalsKernel(const Tensor<2,Scalar,DeviceResident> vertices,
                                     Tensor<2,Scalar,DeviceResident> normals) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;

    if ( x < vertices.dimensionSize(1) / 3) {

        const Eigen::Map<const Vec3> v1(&vertices(0,3*x));
        const Eigen::Map<const Vec3> v2(&vertices(0,3*x+1));
        const Eigen::Map<const Vec3> v3(&vertices(0,3*x+2));

        Vec3 normal = (v3-v1).cross(v2-v1);
        normal.normalize();

        Eigen::Map<Vec3> n1(&normals(0,3*x));
        Eigen::Map<Vec3> n2(&normals(0,3*x+1));
        Eigen::Map<Vec3> n3(&normals(0,3*x+2));

        n1 = normal;
        n2 = normal;
        n3 = normal;

    }

}

template <typename Scalar>
void computeTriangularFaceNormals(const Tensor<2,Scalar,DeviceResident> & vertices,
                                  Tensor<2,Scalar,DeviceResident> & normals) {

    assert(vertices.dimensionSize(0) == 3);
    assert(normals.dimensionSize(0) == 3);

    const int nVertices = vertices.dimensionSize(1);
    assert(normals.dimensionSize(1) == nVertices);

    const uint nFaces = nVertices / 3;

    const dim3 block(512,1,1); // TODO
    const dim3 grid(intDivideAndCeil(nFaces,block.x),1,1);

    computeTriangularFaceNormalsKernel<<<grid,block>>>(vertices,normals);

}

#define COMPUTE_TRIANGULAR_FACE_NORMALS_EXPLICIT_INSTANTIATION(type)                    \
    template void computeTriangularFaceNormals(const Tensor<2,type,DeviceResident> &, \
                                               Tensor<2,type,DeviceResident> &)

ALL_TYPES_INSTANTIATION(COMPUTE_TRIANGULAR_FACE_NORMALS_EXPLICIT_INSTANTIATION);

template <typename Scalar, typename ... NormalizerT>
inline __device__ Eigen::Matrix<Scalar,3,1> normalize(const Eigen::Matrix<Scalar,3,1> & vec, NormalizerT ... normalizer) {

    return vec.normalized();

}

template <typename Scalar>
inline __device__ Eigen::Matrix<Scalar,3,1> normalize(const Eigen::Matrix<Scalar,3,1> & vec, const Eigen::Matrix<Scalar,3,1> & normalizer) {

    return vec.cwiseProduct(normalizer).normalized();

}

template <typename Scalar,
          typename VoxelT,
          typename ... NormalizerT>
__global__ void computeSignedDistanceGradientNormalsKernel(const Tensor<2,Scalar,DeviceResident> vertices,
                                                           Tensor<2,Scalar,DeviceResident> normals,
                                                           Tensor<3,VoxelT,DeviceResident> voxelGrid,
                                                           NormalizerT ... normalizer) {

    typedef Eigen::Matrix<Scalar,3,1> Vec3;

    const uint i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < vertices.dimensionSize(1)) {
        Eigen::Map<Vec3> normalMap(&normals(0,i));

        if (voxelGrid.inBounds(vertices(0,i),vertices(1,i),vertices(2,i),1.f)) {

            if (vertices(0,i) != floor(vertices(0,i))) {
                normalMap = normalize(voxelGrid.transformBackwardGradientValidOnly(SignedDistanceValueExtractor<Scalar,VoxelT>(),
                                                                                   SignedDistanceValidExtractor<Scalar,VoxelT>(),
                                                                                   vertices(0,i),(int)vertices(1,i),(int)vertices(2,i)), normalizer...);
                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),vertices(0,i),(int)vertices(1,i),(int)vertices(2,i))) {
                    printf("whoops!\n");
                }
//                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),vertices(0,i)-1,(int)vertices(1,i),(int)vertices(2,i))) {
//                    printf("whoops!\n");
//                }
//                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),vertices(0,i),(int)vertices(1,i)-1,(int)vertices(2,i))) {
//                    printf("whoops!\n");
//                }
//                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),vertices(0,i),(int)vertices(1,i),(int)vertices(2,i)-1)) {
//                    printf("whoops!\n");
//                }
            } else if (vertices(1,i) != floor(vertices(1,i))) {
                normalMap = normalize(voxelGrid.transformBackwardGradientValidOnly(SignedDistanceValueExtractor<Scalar,VoxelT>(),
                                                                                   SignedDistanceValidExtractor<Scalar,VoxelT>(),
                                                                                   (int)vertices(0,i),vertices(1,i),(int)vertices(2,i)), normalizer...);
                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),(int)vertices(0,i),vertices(1,i),(int)vertices(2,i))) {
                    printf("whoops!\n");
                }
            } else if (vertices(2,i) != floor(vertices(2,i))) {
                normalMap = normalize(voxelGrid.transformBackwardGradientValidOnly(SignedDistanceValueExtractor<Scalar,VoxelT>(),
                                                                                   SignedDistanceValidExtractor<Scalar,VoxelT>(),
                                                                                   (int)vertices(0,i),(int)vertices(1,i),vertices(2,i)), normalizer...);
                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),(int)vertices(0,i),(int)vertices(1,i),vertices(2,i))) {
                    printf("whoops!\n");
                }
            } else {
                normalMap = normalize(voxelGrid.transformBackwardGradientValidOnly(SignedDistanceValueExtractor<Scalar,VoxelT>(),
                                                                                   SignedDistanceValidExtractor<Scalar,VoxelT>(),
                                                                                   (int)vertices(0,i),(int)vertices(1,i),(int)vertices(2,i)), normalizer...);
                if (!voxelGrid.validForInterpolation(SignedDistanceValidExtractor<Scalar,VoxelT>(),(int)vertices(0,i),(int)vertices(1,i),(int)vertices(2,i))) {
                    printf("whoops!\n");
                }
            }
        } else {

            normalMap = Vec3(0,0,0);

        }

    }

}

template <typename Scalar,
          typename VoxelT>
void computeSignedDistanceGradientNormals(const Tensor<2,Scalar,DeviceResident> & vertices,
                                          Tensor<2,Scalar,DeviceResident> & normals,
                                          VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid) {

    typedef Eigen::Matrix<Scalar,3,1> Vec3;

    assert(vertices.dimensionSize(0) == 3);
    assert(normals.dimensionSize(0) == 3);

    const int numVertices = vertices.dimensionSize(1);
    assert(normals.dimensionSize(1) == numVertices);

    if (!numVertices) {
        // there are no points
        return;
    }

    const dim3 block(1024);
    const dim3 grid(intDivideAndCeil((uint)numVertices,block.x));


//    std::cout << normalizer.transpose() << std::endl;
//    std::cout << std::abs(normalizer(0) - normalizer(1)) << std::endl;
//    std::cout << std::abs(normalizer(0) - normalizer(2)) << std::endl;
//    std::cout << std::numeric_limits<Scalar>::epsilon() << std::endl;

    const Vec3 boundingBoxExtent = voxelGrid.boundingBox().max() - voxelGrid.boundingBox().min();

    const Vec3 normalizer = voxelGrid.worldToGridScale();

    const Vec3 voxelSize = boundingBoxExtent.cwiseProduct(normalizer);

//    std::cout << std::abs(boundingBoxExtent(0) - boundingBoxExtent(1)) << std::endl;
//    std::cout << std::abs(boundingBoxExtent(0) - boundingBoxExtent(2)) << std::endl;

    if ( (std::abs(voxelSize(0) - voxelSize(1)) < std::numeric_limits<Scalar>::epsilon()) &&
         (std::abs(voxelSize(0) - voxelSize(2)) < std::numeric_limits<Scalar>::epsilon())) {

        std::cout << "computing isotropic normals" << std::endl;

        computeSignedDistanceGradientNormalsKernel<<<grid,block>>>(vertices,normals,voxelGrid.grid());

    } else {

        std::cout << "computing anisotropic normals" << std::endl;

        computeSignedDistanceGradientNormalsKernel<<<grid,block>>>(vertices,normals,voxelGrid.grid(), normalizer);

    }

}

// TODO: do these really need separate explicit instantiations? can we condense these somehow?
template void computeSignedDistanceGradientNormals(const Tensor<2,float,DeviceResident> &,
                                                   Tensor<2,float,DeviceResident> &,
                                                   VoxelGrid<float,CompositeVoxel<float,TsdfVoxel>,DeviceResident> &);

template void computeSignedDistanceGradientNormals(const Tensor<2,float,DeviceResident> &,
                                                   Tensor<2,float,DeviceResident> &,
                                                   VoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel>,DeviceResident> &);

template void computeSignedDistanceGradientNormals(const Tensor<2,float,DeviceResident> &,
                                                   Tensor<2,float,DeviceResident> &,
                                                   VoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>,DeviceResident> &);


template <typename Scalar, int D>
__global__ void computeVertMapNormalsKernel(const DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > vertMap,
                                            DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > normMap) {

    typedef Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> VecD;

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if ( x < vertMap.dimensionSize(0) && y < vertMap.dimensionSize(1)) {

        const VecD & center = vertMap(x,y);

        if ( (x == 0) || (vertMap(x-1,y)(2) <= Scalar(0)) ) {

            const VecD & right = vertMap(x+1,y);

            if (right(2) <= Scalar(0)) {

                normMap(x,y) = VecD::Zero();

            } else if ( (y == 0) || (vertMap(x,y-1)(2) <= Scalar(0)) ) {

                const VecD & up = vertMap(x,y+1);

                if (up(2) <= Scalar(0)) {

                    normMap(x,y) = VecD::Zero();

                } else {

                    normMap(x,y).template head<3>() =
                            (right.template head<3>() - center.template head<3>()).cross
                            (up.template head<3>() - center.template head<3>()).normalized();

                }

            } else {

                const VecD & down = vertMap(x,y-1);

                normMap(x,y).template head<3>() =
                        (right.template head<3>() - center.template head<3>()).cross
                        (center.template head<3>() - down.template head<3>()).normalized();

            }

        } else {

            const VecD & left = vertMap(x-1,y);

            if ( (y == 0) || (vertMap(x,y-1)(2) <= Scalar(0)) ) {

                const VecD & up = vertMap(x,y+1);

                if (up(2) <= Scalar(0)) {

                    normMap(x,y) = VecD::Zero();

                } else {

                    normMap(x,y).template head<3>() =
                            (center.template head<3>() - left.template head<3>()).cross
                            (up.template head<3>() - center.template head<3>()).normalized();

                }

            } else {

                const VecD & down = vertMap(x,y-1);

                normMap(x,y).template head<3>() =
                        (center.template head<3>() - left.template head<3>()).cross
                        (center.template head<3>() - down.template head<3>()).normalized();

            }

        }

        normMap(x,y) *= Scalar(-1);

    }

}

template <typename Scalar, int D>
void computeVertMapNormals(const DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > & vertMap,
                           DeviceTensor2<Eigen::Matrix<Scalar,D,1,Eigen::DontAlign> > & normMap) {

    const dim3 block(16,8);
    const dim3 grid(intDivideAndCeil(vertMap.dimensionSize(0),block.x),
                    intDivideAndCeil(vertMap.dimensionSize(1),block.y));

    computeVertMapNormalsKernel<<<grid,block>>>(vertMap,normMap);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}

template void computeVertMapNormals(const DeviceTensor2<Eigen::Matrix<float,3,1,Eigen::DontAlign> > & vertMap,
                                    DeviceTensor2<Eigen::Matrix<float,3,1,Eigen::DontAlign> > & normaMap);

template void computeVertMapNormals(const DeviceTensor2<Eigen::Matrix<float,4,1,Eigen::DontAlign> > & vertMap,
                                    DeviceTensor2<Eigen::Matrix<float,4,1,Eigen::DontAlign> > & normaMap);

} // namespace df
