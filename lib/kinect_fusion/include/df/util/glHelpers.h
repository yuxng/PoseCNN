#pragma once

#include <Eigen/Core>

#include <df/voxel/voxelGrid.h>

namespace df {

template <typename Scalar>
inline void glScaleX(const Eigen::Matrix<Scalar,3,1> & scale);

template <>
inline void glScaleX(const Eigen::Vector3f & scale) {
    glScalef(scale(0),scale(1),scale(2));
}

template <>
inline void glScaleX(const Eigen::Vector3d & scale) {
    glScaled(scale(0),scale(1),scale(2));
}

template <typename Scalar>
inline void glTranslateX(const Eigen::Matrix<Scalar,3,1> & translation);

template <>
inline void glTranslateX(const Eigen::Vector3f & translation) {
    glTranslatef(translation(0),translation(1),translation(2));
}

template <>
inline void glTranslateX(const Eigen::Vector3d & translation) {
    glTranslated(translation(0),translation(1),translation(2));
}

template <typename Scalar>
inline void glMultMatrixX(const Eigen::Matrix<Scalar,4,4> & matrix);

template <>
inline void glMultMatrixX(const Eigen::Matrix4f & matrix) {
    glMultMatrixf(matrix.transpose().data());
}

template <>
inline void glMultMatrixX(const Eigen::Matrix4d & matrix) {
    glMultMatrixd(matrix.transpose().data());
}

template <typename Scalar, typename VoxelT, Residency R>
inline void glVoxelGridCoords(const VoxelGrid<Scalar,VoxelT,R> & grid) {

//    typedef Eigen::Matrix<Scalar,3,1> Vec3;

//    const Vec3 & offset = grid.offset();
//    const Vec3 & scale = grid.scale();

//    glTranslateX(offset);
//    glScaleX(scale);

    const Eigen::Matrix<Scalar,4,4> T = grid.gridToWorldTransform();
    glMultMatrixX(T);

}

} // namespace df
