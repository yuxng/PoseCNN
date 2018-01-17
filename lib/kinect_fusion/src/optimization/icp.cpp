#include <df/optimization/icp.h>


#include <df/camera/poly3.h> // TODO

#include <df/optimization/linearSystems.h>

#include <df/util/cudaHelpers.h>

#include <assert.h>

#include <Eigen/Cholesky>

namespace df {

template <typename Scalar,
          typename CameraModelT,
          int DPred,
          typename ... DebugArgsT>
Sophus::SE3<Scalar> icp(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                             const CameraModelT & cameraModel,
                             const Sophus::SE3<Scalar> & predictionPose,
                             const Eigen::Matrix<Scalar,2,1> & depthRange,
                             const Scalar maxError,
                             const uint numIterations,
                             DebugArgsT ... debugArgs) {

    typedef Sophus::SE3<Scalar> SE3;

    const uint width = liveVertices.dimensionSize(0);
    const uint height = liveVertices.dimensionSize(1);

    assert(predVertices.dimensionSize(0) == width);
    assert(predVertices.dimensionSize(1) == height);
    assert(predNormals.dimensionSize(0) == width);
    assert(predNormals.dimensionSize(1) == height);

    const dim3 grid(128,8,1);
    const dim3 block(intDivideAndCeil(width,grid.x),intDivideAndCeil(height,grid.y));

    Eigen::Matrix<Scalar,6,1> initialPose = predictionPose.log();
    Sophus::SE3<Scalar> accumulatedUpdate;

    for (uint iter = 0; iter < numIterations; ++iter) {

        // std::cout << iter << std::endl;

        internal::LinearSystem<Scalar,6> system = internal::icpIteration(liveVertices,
                                                                         predVertices,
                                                                         predNormals,
                                                                         cameraModel,
                                                                         accumulatedUpdate,
                                                                         initialPose,
                                                                         depthRange,maxError,
                                                                         grid,block,
                                                                         debugArgs ...);

        Eigen::Matrix<Scalar,6,6,Eigen::DontAlign> fullJTJ = internal::SquareMatrixReconstructor<Scalar,6>::reconstruct(system.JTJ);

//        Eigen::Matrix<Scalar,6,6> fullJTJ = Eigen::Matrix<Scalar,6,6>::Zero();
//        fullJTJ.template block<1,6>(0,0) = system.JTJ.head;
//        fullJTJ.template block<1,5>(1,1) = system.JTJ.tail.head;
//        fullJTJ.template block<1,4>(2,2) = system.JTJ.tail.tail.head;
//        fullJTJ.template block<1,3>(3,3) = system.JTJ.tail.tail.tail.head;
//        fullJTJ.template block<1,2>(4,4) = system.JTJ.tail.tail.tail.tail.head;
//        fullJTJ.template block<1,1>(5,5) = system.JTJ.tail.tail.tail.tail.tail.head;

//        for (int i=0; i<6; ++i) {
//            for (int j=0; j<6; ++j) {
//                fullJTJ(j,i) = fullJTJ(i,j);
//            }
//        }

//        for (int i = 0; i < 6; ++i) {
//            system.JTr(i) = 0;
//        }

//        std::cout << fullJTJ << std::endl;

//        std::cout << std::endl << system.JTr << std::endl << std::endl << std::endl;

//        Eigen::Matrix<Scalar,6,6> sav = fullJTJ.template selfadjointView<Eigen::Upper>();
//        std::cout << sav << std::endl;
//        Eigen::Matrix<Scalar,6,1> solution = sav.ldlt().solve(system.JTr);
//        std::cout << solution << std::endl;

        Eigen::Matrix<Scalar,6,1> solution = fullJTJ.template selfadjointView<Eigen::Upper>().ldlt().solve(system.JTr);
//        std::cout << std::endl << solution2 << std::endl;

        // std::cout << fullJTJ << std::endl;

        // std::cout << system.JTr << std::endl;

        // std::cout << solution.transpose() << std::endl;

        SE3 update = SE3::exp(solution);
        accumulatedUpdate = update*accumulatedUpdate;

//        std::cout << accumulatedUpdate.matrix() << std::endl << std::endl;

    }

    return accumulatedUpdate;

}

template Sophus::SE3f icp(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint);

template Sophus::SE3f icp(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint,
                          DeviceTensor2<Eigen::UnalignedVec4<uchar> >);


template Sophus::SE3f icp(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint);

template Sophus::SE3f icp(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint,
                          DeviceTensor2<Eigen::UnalignedVec4<uchar> >);


} // namespace df
