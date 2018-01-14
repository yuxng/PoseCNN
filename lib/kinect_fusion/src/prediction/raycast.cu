#include <df/prediction/raycast.h>

#include <assert.h>

#include <df/util/cudaHelpers.h>

namespace df {

template <typename Scalar, uint D>
__device__ inline Eigen::Matrix<Scalar,D,1> componentwiseMin(const Eigen::Matrix<Scalar,D,1> & a,
                                                             const Eigen::Matrix<Scalar,D,1> & b) {

    const Eigen::Map<const Eigen::Array<Scalar,D,1> > aMap(a.data());
    const Eigen::Map<const Eigen::Array<Scalar,D,1> > bMap(b.data());
    return aMap.min(bMap);

}

template <typename Scalar, uint D>
__device__ inline Eigen::Matrix<Scalar,D,1> componentwiseMax(const Eigen::Matrix<Scalar,D,1> & a,
                                                             const Eigen::Matrix<Scalar,D,1> & b) {

    const Eigen::Map<const Eigen::Array<Scalar,D,1> > aMap(a.data());
    const Eigen::Map<const Eigen::Array<Scalar,D,1> > bMap(b.data());
    return aMap.max(bMap);

}

template <typename Scalar>
__device__ inline Eigen::Matrix<Scalar,2,1> boxIntersections(const Eigen::Matrix<Scalar,3,1> & rayOrigin,
                                                             const Eigen::Matrix<Scalar,3,1> & rayDirection,
                                                             const Eigen::Matrix<Scalar,3,1> & boxMin,
                                                             const Eigen::Matrix<Scalar,3,1> & boxMax) {

    typedef Eigen::Matrix<Scalar,2,1> Vec2;
    typedef Eigen::Matrix<Scalar,3,1> Vec3;

    const Vec3 inverseRayDirection = rayDirection.unaryExpr([](const Scalar val){ return Scalar(1) / val; });

    const Vec3 boxMinIntersections = inverseRayDirection.cwiseProduct(boxMin - rayOrigin);

    const Vec3 boxMaxIntersections = inverseRayDirection.cwiseProduct(boxMax - rayOrigin);

    const Vec3 minTimeIntersections = componentwiseMin<Scalar,3>(boxMinIntersections,boxMaxIntersections);

    const Vec3 maxTimeIntersections = componentwiseMax<Scalar,3>(boxMinIntersections,boxMaxIntersections);

    const Scalar maximalEntranceTime = minTimeIntersections.maxCoeff();

    const Scalar minimalExitTime = maxTimeIntersections.minCoeff();

    return Vec2(maximalEntranceTime,minimalExitTime);

}

template <typename Scalar,
          typename VoxelT,
          typename CameraModelT>
__global__ void raycastKernel(Tensor<3,Scalar,DeviceResident> predictedVertices,
                              Tensor<3,Scalar,DeviceResident> predictedNormals,
                              const VoxelGrid<Scalar,VoxelT,DeviceResident> voxelGrid, // TODO: if we dont end up using scale/offset, we can just pass in the Tensor portion
                              const CameraModelT cameraModel,
                              const Sophus::SE3<Scalar> transformWorldToPrediction, // TODO: faster to pass in both or do in-kernel?
                              const Sophus::SE3<Scalar> transformPredictionToWorld) {

    typedef Eigen::Matrix<Scalar,2,1> Vec2;
    typedef Eigen::Matrix<Scalar,3,1> Vec3;

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = threadIdx.y + blockDim.y * blockIdx.y;

    if ( (x < predictedVertices.dimensionSize(1)) && (y < predictedVertices.dimensionSize(2)) ) {

        const Vec2 pixel(x,y);

        const Vec3 predictionRayDirection = cameraModel.unproject(pixel,Scalar(1)).normalized();

        // TODO: can we change the frame some of the computation is done in to eliminate the need for both transforms?
        const Vec3 worldRayOrigin = transformPredictionToWorld.translation();

        const Vec3 worldRayDirection = transformPredictionToWorld.rotationMatrix()*predictionRayDirection;

        const Vec3 gridRayOrigin = voxelGrid.worldToGrid(worldRayOrigin);

        const Vec3 & gridRayDirection = worldRayDirection;

        const Vec3 volumeMin = Vec3(0,0,0);

        const Vec3 volumeMax = voxelGrid.dimensions().template cast<Scalar>() - Vec3(1,1,1); //voxelGrid.max();

        const Vec2 volumeEntranceExit = boxIntersections(gridRayOrigin,gridRayDirection,
                                                         volumeMin,volumeMax);

        Eigen::Map<Vec3> predictedVertex(&predictedVertices(0,x,y));
        Eigen::Map<Vec3> predictedNormal(&predictedNormals(0,x,y));

        if (volumeEntranceExit(0) > volumeEntranceExit(1)) {
            // the ray does not enter the volume at any time
            predictedVertex(2) = 0;
            return;
        }

        Scalar currentT = max(Scalar(0),volumeEntranceExit(0));

        while (currentT < volumeEntranceExit(1)) {

            const Vec3 currentPoint = gridRayOrigin + currentT * gridRayDirection;

//            const VoxelT interpolatedVoxel = voxelGrid.grid().interpolate(currentPoint(0),currentPoint(1),currentPoint(2));



        }

        predictedVertex = worldRayOrigin + volumeEntranceExit(1) * worldRayDirection;
        predictedNormal = Vec3(0,0,-1);

    }

}

template <typename Scalar,
          typename VoxelT,
          typename CameraModelT>
void raycast(Tensor<3,Scalar,DeviceResident> & predictedVertices,
             Tensor<3,Scalar,DeviceResident> & predictedNormals,
             const VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid,
             const CameraModelT & cameraModel,
             const Sophus::SE3<Scalar> & transformWorldToPrediction) {

    const uint width = predictedVertices.dimensionSize(1);
    const uint height = predictedVertices.dimensionSize(2);
    assert(predictedVertices.dimensionSize(0) == 3);
    assert(width == predictedNormals.dimensionSize(1));
    assert(height == predictedNormals.dimensionSize(2));
    assert(predictedNormals.dimensionSize(0) == 3);

    const dim3 block(32,16,1);
    const dim3 grid(intDivideAndCeil(width,block.x),intDivideAndCeil(height,block.y),1);

    raycastKernel<<<grid,block>>>(predictedVertices,predictedNormals,
                                  voxelGrid,cameraModel,
                                  transformWorldToPrediction,
                                  transformWorldToPrediction.inverse());

}

} // namespace df

#include <df/camera/poly3.h>
#include <df/voxel/tsdf.h>

namespace df {

template void raycast(Tensor<3,float,DeviceResident> &,
                      Tensor<3,float,DeviceResident> &,
                      const VoxelGrid<float,TsdfVoxel,DeviceResident> &,
                      const Poly3CameraModel<float> &,
                      const Sophus::SE3f &);


} // namespace df
