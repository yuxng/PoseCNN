#include <df/optimization/icp.h>

#include <df/optimization/linearSystems.h>

#include <df/util/cudaHelpers.h>
#include <df/util/eigenHelpers.h>

#include <df/camera/poly3.h> // TODO

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <df/util/globalTimer.h>

#include <Eigen/Core>

namespace df {

template <typename Scalar,
          typename CameraModelT,
          int DPred>
__global__ void icpKernel(internal::JacobianAndResidual<Scalar,1,6> * jacobiansAndResiduals,
                          const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > liveVertices,
                          const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > predictedVertices,
                          const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > predictedNormals,
                          const CameraModelT cameraModel,
                          const Sophus::SE3Group<Scalar> updatedPose,
                          const Eigen::Matrix<Scalar,2,1> depthRange,
                          const Scalar maxError) {

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

//        if ( x > 200 && x < 220 && y > 200 && y < 220) {
//            printf("(%d,%d) -> %f\n",x,y,predDepth);
//        }

//        if (x == 0 && y == 0) {
//            internal::LinearSystemCreationFunctor<Scalar,1,6> theCreator;
//            for (int i=0 ; i< 6; ++i) {
//                jacobiansAndResiduals[0].J(i) = i+1;
//            }
//            jacobiansAndResiduals[0].r = 0.5;
//            internal::LinearSystem<Scalar,6> system = theCreator(jacobiansAndResiduals[0]);
//            printf("^$ %f %f %f %f %f %f\n",system.JTr(0),system.JTr(1),system.JTr(2),
//                                            system.JTr(3),system.JTr(4),system.JTr(5));

//            printf("\n %f %f %f %f %f %f\n",system.JTJ.head(0),system.JTJ.head(1),system.JTJ.head(2),system.JTJ.head(3),system.JTJ.head(4),system.JTJ.head(5));
//            printf("\n       %f %f %f %f %f\n",system.JTJ.tail.head(0),system.JTJ.tail.head(1),system.JTJ.tail.head(2),system.JTJ.tail.head(3),system.JTJ.tail.head(4));
//            printf("\n             %f %f %f %f\n",system.JTJ.tail.tail.head(0),system.JTJ.tail.tail.head(1),system.JTJ.tail.tail.head(2),system.JTJ.tail.tail.head(3));
//            printf("\n                   %f %f %f\n",system.JTJ.tail.tail.tail.head(0),system.JTJ.tail.tail.tail.head(1),system.JTJ.tail.tail.tail.head(2));
//            printf("\n                         %f %f\n",system.JTJ.tail.tail.tail.tail.head(0),system.JTJ.tail.tail.tail.tail.head(1));
//            printf("\n                               %f\n",system.JTJ.tail.tail.tail.tail.tail.head(0));

//            internal::LinearSystemSumFunctor<Scalar,6> theSummer;
//            system = theSummer(system,system);

//            printf("^$ %f %f %f %f %f %f\n",system.JTr(0),system.JTr(1),system.JTr(2),
//                                            system.JTr(3),system.JTr(4),system.JTr(5));

//            printf("\n %f %f %f %f %f %f\n",system.JTJ.head(0),system.JTJ.head(1),system.JTJ.head(2),system.JTJ.head(3),system.JTJ.head(4),system.JTJ.head(5));
//            printf("\n       %f %f %f %f %f\n",system.JTJ.tail.head(0),system.JTJ.tail.head(1),system.JTJ.tail.head(2),system.JTJ.tail.head(3),system.JTJ.tail.head(4));
//            printf("\n             %f %f %f %f\n",system.JTJ.tail.tail.head(0),system.JTJ.tail.tail.head(1),system.JTJ.tail.tail.head(2),system.JTJ.tail.tail.head(3));
//            printf("\n                   %f %f %f\n",system.JTJ.tail.tail.tail.head(0),system.JTJ.tail.tail.tail.head(1),system.JTJ.tail.tail.tail.head(2));
//            printf("\n                         %f %f\n",system.JTJ.tail.tail.tail.tail.head(0),system.JTJ.tail.tail.tail.tail.head(1));
//            printf("\n                               %f\n",system.JTJ.tail.tail.tail.tail.tail.head(0));

//        }

        if ((predictedDepth > depthRange(0)) && predictedDepth < depthRange(1)) {

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

            if ( (u > border) && (u < (width-1-border)) && (v > border) && (v < (height-1-border)) ) {

                const Vec3 & liveVertex = liveVertices(u,v);

                const Scalar liveDepth = liveVertex(2);

                if ((liveDepth > depthRange(0)) && (liveDepth < depthRange(1))) {

                    // TODO: double-check validity of this method of getting the ray
                    const Vec3 ray = updatedPredVertex.normalized();

                    const VecD & predictedNormal = predictedNormals(x,y);

                    if (-ray.dot(predictedNormal.template head<3>()) > rayNormDotThreshold) {

                        const Scalar error = predictedNormal.template head<3>().dot(liveVertex - updatedPredVertex);

                        if (error < maxError) {

                            const Scalar weightSqrt = Scalar(1) / (liveDepth);

                            const Eigen::Matrix<Scalar,1,3> dError_dUpdatedPredictedPoint = predictedNormal.template head<3>().transpose();
                            Eigen::Matrix<Scalar,3,6> dUpdatedPredictedPoint_dUpdate;
                            dUpdatedPredictedPoint_dUpdate << 1, 0, 0,                     0,  updatedPredVertex(2), -updatedPredVertex(1),
                                                              0, 1, 0, -updatedPredVertex(2),                     0,  updatedPredVertex(0),
                                                              0, 0, 1,  updatedPredVertex(1), -updatedPredVertex(0),                     0;

                            jacobiansAndResiduals[x + width*y].J = weightSqrt*dError_dUpdatedPredictedPoint*dUpdatedPredictedPoint_dUpdate;
                            jacobiansAndResiduals[x + width*y].r = weightSqrt*error;

                        }

                    }

                }
            }

        }

    }

}

namespace internal {

template <typename Scalar,
          typename CameraModelT,
          int DPred>
LinearSystem<Scalar,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                                    const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                                    const CameraModelT & cameraModel,
                                    const Sophus::SE3Group<Scalar> & predictionPose,
                                    const Eigen::Matrix<Scalar,2,1> & depthRange,
                                    const Scalar maxError,
                                    const dim3 grid,
                                    const dim3 block) {

    // TODO: make efficient
    static thrust::device_vector<JacobianAndResidual<Scalar,1,6> > jacobiansAndResiduals(liveVertices.count());

    GlobalTimer::tick("icpKernel");
    cudaFuncSetCacheConfig(icpKernel<Scalar,CameraModelT,DPred>, cudaFuncCachePreferL1);
    icpKernel<Scalar><<<grid,block>>>(thrust::raw_pointer_cast(jacobiansAndResiduals.data()),
                                      liveVertices,predVertices,predNormals,
                                      cameraModel,
                                      predictionPose,
                                      depthRange,
                                      maxError);

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
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3);

template LinearSystem<float,6> icpIteration(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                                            const Poly3CameraModel<float> &,
                                            const Sophus::SE3f &,
                                            const Eigen::Vector2f &,
                                            const float,
                                            const dim3, const dim3);

} // namespace internal


} // namespace df
