#pragma once

#include <pangolin/pangolin.h>

#include <df/camera/camera.h>

namespace df {

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<Scalar> & camera,
                                                       const Scalar zNear, const Scalar zFar);

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<Scalar> & camera,
                                                          const Scalar zNear, const Scalar zFar);

} // namespace df
