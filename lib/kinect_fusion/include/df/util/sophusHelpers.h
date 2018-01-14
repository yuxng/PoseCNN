#pragma once

#include <sophus/se3.hpp>

namespace df {

namespace operators {

template <typename Scalar>
inline std::ostream & operator<<(std::ostream & stream, const Sophus::SE3<Scalar> & transform) {

    stream << transform.matrix();
    return stream;

}

} // namespace operators


} // namespace df
