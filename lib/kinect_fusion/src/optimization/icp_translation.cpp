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
Sophus::SE3Group<Scalar> icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<Scalar> > & liveVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predVertices,
                             const DeviceTensor2<Eigen::UnalignedVec<Scalar,DPred> > & predNormals,
                             const CameraModelT & cameraModel,
                             const Sophus::SE3Group<Scalar> & predictionPose,
                             const Eigen::Matrix<Scalar,2,1> & depthRange,
                             const Scalar maxError,
                             const uint numIterations,
                             DebugArgsT ... debugArgs) {

    typedef Sophus::SE3Group<Scalar> SE3;

    const uint width = liveVertices.dimensionSize(0);
    const uint height = liveVertices.dimensionSize(1);

    assert(predVertices.dimensionSize(0) == width);
    assert(predVertices.dimensionSize(1) == height);
    assert(predNormals.dimensionSize(0) == width);
    assert(predNormals.dimensionSize(1) == height);

    const dim3 grid(128,8,1);
    const dim3 block(intDivideAndCeil(width,grid.x),intDivideAndCeil(height,grid.y));

    Eigen::Matrix<Scalar,3,1> initialTranslation = predictionPose.translation();
    Sophus::SE3Group<Scalar> accumulatedUpdate;

    for (uint iter = 0; iter < numIterations; ++iter) {

        std::cout << iter << std::endl;

        internal::LinearSystem<Scalar,3> system = internal::icpIteration_translation(liveVertices,
                                                                         predVertices,
                                                                         predNormals,
                                                                         cameraModel,
                                                                         accumulatedUpdate,
                                                                         initialTranslation,
                                                                         depthRange,maxError,
                                                                         grid,block,
                                                                         debugArgs ...);

        Eigen::Matrix<Scalar,3,3,Eigen::DontAlign> fullJTJ = internal::SquareMatrixReconstructor<Scalar,3>::reconstruct(system.JTJ);

        Eigen::Matrix<Scalar,3,1> solution = fullJTJ.template selfadjointView<Eigen::Upper>().ldlt().solve(system.JTr);

        std::cout << fullJTJ << std::endl;

        std::cout << system.JTr << std::endl;

        std::cout << solution.transpose() << std::endl;

        Eigen::Matrix<Scalar,6,1> solution_full;
        solution_full(0) = 0;
        solution_full(1) = 0;
        solution_full(2) = 0;
        solution_full(3) = solution(0);
        solution_full(4) = solution(1);
        solution_full(5) = solution(2);

        SE3 update = SE3::exp(solution_full);
        accumulatedUpdate = update*accumulatedUpdate;

        std::cout << accumulatedUpdate.matrix() << std::endl << std::endl;

    }

    return accumulatedUpdate;

}

template Sophus::SE3f icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint);

template Sophus::SE3f icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint,
                          DeviceTensor2<Eigen::UnalignedVec4<uchar> >);


template Sophus::SE3f icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint);

template Sophus::SE3f icp_translation(const DeviceTensor2<Eigen::UnalignedVec3<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const DeviceTensor2<Eigen::UnalignedVec4<float> > &,
                          const Poly3CameraModel<float> &,
                          const Sophus::SE3f &,
                          const Eigen::Vector2f &,
                          const float,
                          const uint,
                          DeviceTensor2<Eigen::UnalignedVec4<uchar> >);


} // namespace df
