#include <df/image/imagePyramid.h>

#include <df/image/bilateralFilter.h>

namespace df {

template <typename Scalar, Residency R>
std::size_t ImagePyramid<Scalar,R>::sizeRequired(const int width, const int height, const int nLevels) {

    return width*height + (nLevels > 1 ? sizeRequired(width/2,height/2,nLevels-1) : 0);

}

template <typename Scalar, Residency R>
void ImagePyramid<Scalar,R>::computePyramid(DeviceTensor1<Scalar> & halfKernel, const Scalar sigmaInput) {

    for (int i = 1; i < images_.size(); ++i) {

        radiallySymmetricBilateralFilterAndDownsampleBy2(images_[i],images_[i-1],halfKernel,sigmaInput);

    }

}

template class ImagePyramid<float,DeviceResident>;

} // namespace df
