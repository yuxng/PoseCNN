#pragma once

#include <df/optimization/linearSystems.h>
#include <df/transform/nonrigid.h>

#include <df/voxel/voxelGrid.h>

#include <Eigen/Sparse>

namespace df {

template <typename Scalar, typename ScalarOpt, template <typename,int...> class TransformT, internal::TransformUpdateMethod U>
void computeRegularizerNormalEquations(const NonrigidTransformer<Scalar,TransformT> & transformer,
                                       Eigen::SparseMatrix<ScalarOpt> & JTJ,
                                       Eigen::Matrix<ScalarOpt,Eigen::Dynamic,1> & Jtr);



} // namespace df
