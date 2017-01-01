#pragma once

#include <vector>

#include <df/util/tensor.h>

namespace df {

template <typename Scalar, Residency R>
class ImagePyramid {
public:

    ImagePyramid(const int width, const int height, const int nLevels)
        : data_( sizeRequired(width, height, nLevels) ) {

        images_.emplace_back(Eigen::Matrix<uint,2,1>(width,height),data_.data());

        for (int i = 1; i < nLevels; ++i) {

            images_.emplace_back(Eigen::Matrix<uint,2,1>(width >> i, height >> i),
                                 images_[i-1].data() + images_[i-1].count());

        }

    }

    inline Tensor<2,Scalar,R> & image(const int level) {

        return images_[level];

    }

    void computePyramid(DeviceTensor1<Scalar> & halfKernel, const Scalar sigmaInput);

private:

    static std::size_t sizeRequired(const int width, const int height, const int nLevels);

    std::vector<Tensor<2,Scalar,R> > images_;
    ManagedTensor<1,Scalar,R> data_;

};

template <typename Scalar>
using DeviceImagePyramid = ImagePyramid<Scalar,DeviceResident>;

} // namespace df
