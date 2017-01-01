#include <df/util/pangolinHelpers.h>

#include <df/util/macros.h>

namespace df {

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<Scalar> & camera,
                                                       const Scalar zNear, const Scalar zFar) {

    return pangolin::ProjectionMatrixRDF_TopLeft(camera.width(),camera.height(),
                                                 camera.params()[0],camera.params()[1],
                                                 camera.params()[2],camera.params()[3],
                                                 zNear, zFar);

}

#define PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(type)                           \
    template pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<type> &, \
                                                                    const type, const type)

PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(float);
PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(double);



template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<Scalar> & camera,
                                                          const Scalar zNear, const Scalar zFar) {

    return pangolin::ProjectionMatrixRDF_BottomLeft(camera.width(),camera.height(),
                                                    camera.params()[0],camera.params()[1],
                                                    camera.params()[2],camera.params()[3],
                                                    zNear, zFar);

}

#define PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(type)                           \
    template pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<type> &, \
                                                                    const type, const type)

PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(float);
PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(double);


} // namespace df
