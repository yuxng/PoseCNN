#pragma once

#include <df/util/nanoflannHelpers.h>
#include <df/util/tensor.h>

namespace df {

template <typename Scalar>
uint decimate(const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
              HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & decimatedVertices,
              const Scalar radius,
              const int initialIndex = 0);

template <typename Scalar>
uint decimate(const HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
              df::KDTree<Scalar> & tree,
              HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & decimatedVertices,
              const Scalar radius,
              const int initialIndex = 0);

template <typename Scalar>
uint decimateIncremental(const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & originalVertices,
                         const ConstHostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & newVertices,
                         HostTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & newDecimatedVertices,
                         const Scalar radius);

} // namespace df
