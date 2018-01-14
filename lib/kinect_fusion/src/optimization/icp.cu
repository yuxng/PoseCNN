#include <df/camera/poly3.h> // TODO

#include <df/optimization/icp.h>
#include <df/optimization/linearSystems.h>

#include <df/util/cudaHelpers.h>
#include <df/util/debugHelpers.h>
#include <df/util/eigenHelpers.h>
#include <df/util/globalTimer.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <Eigen/Core>

namespace df {


template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
__global__ void icpKernel(internal::JacobianAndResidual<Scalar,1,6> * jacobiansAndResiduals,
                          const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > liveVertices,
                          const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > predictedVertices,
                          const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > predictedNormals,
                          const CameraModelT cameraModel,
                          const Sophus::SE3<Scalar> updatedPose,
                          const Eigen::Matrix<Scalar,6,1> initialPose,
                          const Eigen::Matrix<Scalar,2,1> depthRange,
                          const Scalar maxError,
                          DebugArgsT ... debugArgs) {

    typedef Eigen::Matrix<Scalar,DPred,1,Eigen::DontAlign> VecD;
    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<Scalar,2,1,Eigen::DontAlign> Vec2;

    static constexpr Scalar border = Scalar(2); // TODO
    static constexpr Scalar rayNormDotThreshold = Scalar(0.1); // TODO

    const uint x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint y = threadIdx.y + blockIdx.y * blockDim.y;

    const uint width = liveVertices.dimensionSize(0);
    const uint height = liveVertices.dimensionSize(1);

    // TODO: template for guaranteed in-bound blocking
    if (x < width && y < height) {

        // TODO: take care of this with a memset?
        jacobiansAndResiduals[x + width*y].J = Eigen::Matrix<Scalar,1,6>::Zero();
        jacobiansAndResiduals[x + width*y].r = 0;

        const VecD & predictedVertex = predictedVertices(x,y);

        const Scalar predictedDepth = predictedVertex(2);

        if ((predictedDepth < depthRange(0)) || predictedDepth > depthRange(1)) {

            PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,255,0,255),debugArgs...);

            return;
        }

        const Vec3 updatedPredVertex = updatedPose * predictedVertex.template head<3>();

        const Vec2 projectedPredVertex = cameraModel.project(updatedPredVertex);

        //            const Vec2 projectedPredVertex  (updatedPredVertex(0)/updatedPredVertex(2)*cameraModel.params()[0] + cameraModel.params()[2],
        //                                             updatedPredVertex(1)/updatedPredVertex(2)*cameraModel.params()[1] + cameraModel.params()[3]);
        //            if ( x > 200 && x < 220 && y > 200 && y < 220) {
        //                printf("(%d,%d) -> (%f,%f)\n",x,y,projectedPredVertex(0),projectedPredVertex(1));
        //            }

        // TODO: interpolate?
        const int u = projectedPredVertex(0) + Scalar(0.5);
        const int v = projectedPredVertex(1) + Scalar(0.5);

        if ( (u <= border) || (u >= (width-1-border)) || (v <= border) || (v >= (height-1-border)) ) {

            PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,0,255,255),debugArgs...);
            return;

        }

        const Vec3 & liveVertex = liveVertices(u,v);

        const Scalar liveDepth = liveVertex(2);

        if ((liveDepth < depthRange(0)) || (liveDepth > depthRange(1))) {

            PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,0,255,255),debugArgs...);
            return;

        }

        // TODO: double-check validity of this method of getting the ray
        const Vec3 ray = updatedPredVertex.normalized();

        const VecD & predictedNormal = predictedNormals(x,y);

        if (-ray.dot(predictedNormal.template head<3>()) < rayNormDotThreshold) {

            PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(255,0,0,255),debugArgs...);
            return;

        }

        const Scalar error = predictedNormal.template head<3>().dot(liveVertex - updatedPredVertex);

        const Scalar absError = fabs(error);

        if (absError > maxError) {

            PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(0,255,0,255),debugArgs...);
            return;

        }

        const Scalar weightSqrt = Scalar(1) / (liveDepth);

        const Eigen::Matrix<Scalar,1,3> dError_dUpdatedPredictedPoint = predictedNormal.template head<3>().transpose();
        Eigen::Matrix<Scalar,3,6> dUpdatedPredictedPoint_dUpdate;
        dUpdatedPredictedPoint_dUpdate << 1, 0, 0,                     0,  updatedPredVertex(2), -updatedPredVertex(1),
                                          0, 1, 0, -updatedPredVertex(2),                     0,  updatedPredVertex(0),
                                          0, 0, 1,  updatedPredVertex(1), -updatedPredVertex(0),                     0;

        jacobiansAndResiduals[x + width*y].J = weightSqrt * dError_dUpdatedPredictedPoint * dUpdatedPredictedPoint_dUpdate;
        jacobiansAndResiduals[x + width*y].r = weightSqrt * error;

        const uchar gray = min(Scalar(255),255 * absError / maxError );
        PixelDebugger<DebugArgsT...>::debugPixel(Eigen::Vector2i(x,y),Eigen::UnalignedVec4<uchar>(gray,gray,gray,255),debugArgs...);

    }

}

namespace internal {

template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
LinearSystem<Scalar,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                                    const CameraModelT & cameraModel,
                                    const Sophus::SE3<Scalar> & predictionPose,
                                    const Eigen::Matrix<Scalar,6,1>& initialPose,
                                    const Eigen::Matrix<Scalar,2,1> & depthRange,
                                    const Scalar maxError,
                                    const dim3 grid,
                                    const dim3 block,
                                    DebugArgsT ... debugArgs) {

    // TODO: make efficient
    static thrust::device_vector<JacobianAndResidual<Scalar,1,6> > jacobiansAndResiduals(liveVertices.count());

    GlobalTimer::tick("icpKernel");
    cudaFuncSetCacheConfig(icpKernel<Scalar,CameraModelT,DPred>, cudaFuncCachePreferL1);
    icpKernel<Scalar><<<grid,block>>>(thrust::raw_pointer_cast(jacobiansAndResiduals.data()),
                                      liveVertices,predVertices,predNormals,
                                      cameraModel,
                                      predictionPose,
                                      initialPose,
                                      depthRange,
                                      maxError,
                                      debugArgs ...);

    cudaDeviceSynchronize();
    CheckCudaDieOnError();
    GlobalTimer::tock("icpKernel");

//    static thrust::device_vector<LinearSystem<Scalar,6> > systems(jacobiansAndResiduals.size());

//    std::cout << sizeof(LinearSystem<Scalar,6>) << std::endl;
//    std::cout << sizeof(LinearSystem2<Scalar,6>) << std::endl;

//    std::cout << sizeof(RawVec<Scalar,6*7/2>) << std::endl;
//    std::cout << sizeof(RawVec<Scalar,1>) << std::endl;
//    std::cout << sizeof(LinearSystem3<Scalar,6>) << std::endl;

//    GlobalTimer::tick("transform");
//    thrust::transform(jacobiansAndResiduals.begin(),jacobiansAndResiduals.end(),
//                      systems.begin(),LinearSystemCreationFunctor<Scalar,1,6>());
//    cudaDeviceSynchronize();
//    CheckCudaDieOnError();

//    GlobalTimer::tock("transform");
//    GlobalTimer::tick("reduce");
//    LinearSystem<Scalar,6> system = thrust::reduce(systems.begin(),systems.end(),LinearSystem<Scalar,6>::zero(),LinearSystemSumFunctor<Scalar,6>());
//    cudaDeviceSynchronize();
//    GlobalTimer::tock("reduce");

//    CheckCudaDieOnError();


//    GlobalTimer::tick("transform_reduce");
//    LinearSystem2<Scalar,6> system = thrust::transform_reduce(jacobiansAndResiduals.begin(),
//                                                             jacobiansAndResiduals.end(),
//                                                             LinearSystemCreationFunctor2<Scalar,1,6>(),
//                                                             LinearSystem2<Scalar,6>::zero(),
//                                                             LinearSystemSumFunctor2<Scalar,6>());

//    cudaDeviceSynchronize();
//    CheckCudaDieOnError();
//    GlobalTimer::tock("transform_reduce");

    GlobalTimer::tick("transform_reduce");
    LinearSystem<Scalar,6> system = thrust::transform_reduce(jacobiansAndResiduals.begin(),
                                                             jacobiansAndResiduals.end(),
                                                             LinearSystemCreationFunctor<Scalar,1,6>(),
                                                             LinearSystem<Scalar,6>::zero(),
                                                             LinearSystemSumFunctor<Scalar,6>());
/*
    static constexpr Scalar huberAlpha = Scalar(0.01);

    const Scalar totalResidual = thrust::transform_reduce(jacobiansAndResiduals.begin(),
                                                          jacobiansAndResiduals.end(),
                                                          ResidualFunctorHuber<Scalar,1,6>(huberAlpha),
                                                          Scalar(0),
                                                          thrust::plus<Scalar>());

    std::cout << "icp residual" << totalResidual << std::endl;



    GlobalTimer::tick("transform_reduce");
    LinearSystem<Scalar,6> system = thrust::transform_reduce(jacobiansAndResiduals.begin(),
                                                             jacobiansAndResiduals.end(),
                                                             LinearSystemCreationFunctorHuber<Scalar,1,6>(huberAlpha),
                                                             LinearSystem<Scalar,6>::zero(),
                                                             LinearSystemSumFunctor<Scalar,6>());
*/

    cudaDeviceSynchronize();
    CheckCudaDieOnError();
    GlobalTimer::tock("transform_reduce");

//    std::cout << "size: " << sizeof(LinearSystem<Scalar,6>) << std::endl;

//    LinearSystem2<Scalar,6> * sysptr = reinterpret_cast<LinearSystem2<Scalar,6> *>(&system);
//    return *sysptr;


    return system;

}


template LinearSystem<float,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const Poly3CameraModel<float> &,
                                            const Sophus::SE3f &,
                                            const Eigen::Matrix<float,6,1> &,
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3);

template LinearSystem<float,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const Poly3CameraModel<float> &,
                                            const Sophus::SE3f &,
                                            const Eigen::Matrix<float,6,1> &,
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3,                                                                                         \
                                            DeviceTensor2<Eigen::UnalignedVec4<uchar> >);

template LinearSystem<float,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const Poly3CameraModel<float> &,
                                            const Sophus::SE3f &,
                                            const Eigen::Matrix<float,6,1> &,
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3);

template LinearSystem<float,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const Poly3CameraModel<float> &,
                                            const Sophus::SE3f &,
                                            const Eigen::Matrix<float,6,1> &,
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3,                                                                                         \
                                            DeviceTensor2<Eigen::UnalignedVec4<uchar> >);

} // namespace internal


} // namespace df
