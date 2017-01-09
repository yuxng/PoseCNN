#include <df/image/bilateralFilter.h>

#include <df/util/cudaHelpers.h>

#include <assert.h>

namespace df {

template <typename Scalar, typename ScalarComp>
__global__ void radiallySymmetricBilateralFilterKernel(DeviceTensor2<Scalar> destination, const DeviceTensor2<Scalar> source,
                                                       DeviceTensor1<ScalarComp> halfKernel, const ScalarComp sigmaInput) {

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    // TODO: remove for guaranteed inbounds
    if (x < destination.width() && y < destination.height()) {

        const Scalar center = source(x,y);

        ScalarComp totalWeight(0);
        ScalarComp totalValue(0);

        const int window = halfKernel.length() - 1;

        const int dxMin = max(-window, -x);
        const int dxMax = min( window, destination.width() - x - 1);

        const int dyMin = max(-window, -y);
        const int dyMax = min( window, destination.height() - y - 1);

        for (int dy = dyMin; dy <= dyMax; ++dy) {
            for (int dx = dxMin; dx <= dxMax; ++dx) {

                const Scalar sourceVal = source(x + dx, y + dy);

                const Scalar diff = sourceVal - center;

                if (diff < 3 * sigmaInput) {

                    const Scalar weight = halfKernel(abs(dx))*halfKernel(abs(dy)) * expf( -(diff * diff) / (2 * sigmaInput * sigmaInput) );

                    totalWeight += weight;

                    totalValue += weight * sourceVal;

                }

            }
        }

        destination(x,y) = totalValue / totalWeight;

    }

}

template <typename Scalar, typename ScalarComp>
void radiallySymmetricBilateralFilter(DeviceTensor2<Scalar> & destination, const DeviceTensor2<Scalar> & source,
                                      DeviceTensor1<ScalarComp> & halfKernel, const ScalarComp sigmaInput) {

    const dim3 block(16,16);
    const dim3 grid(intDivideAndCeil(source.width(),block.x),
                    intDivideAndCeil(source.height(),block.y));

    radiallySymmetricBilateralFilterKernel<<<grid,block>>>(destination,source,halfKernel,sigmaInput);


}

template void radiallySymmetricBilateralFilter(DeviceTensor2<ushort> & destination, const DeviceTensor2<ushort> & source,
                                               DeviceTensor1<float> & halfKernel, const float sigmaInput);

template void radiallySymmetricBilateralFilter(DeviceTensor2<float> & destination, const DeviceTensor2<float> & source,
                                               DeviceTensor1<float> & halfKernel, const float sigmaInput);





template <typename Scalar, typename ScalarComp>
__global__ void radiallySymmetricBilateralFilterAndDownsampleBy2Kernel(DeviceTensor2<Scalar> destination, const DeviceTensor2<Scalar> source,
                                                                       DeviceTensor1<ScalarComp> halfKernel, const ScalarComp sigmaInput) {

    const int xDst = threadIdx.x + blockDim.x * blockIdx.x;
    const int yDst = threadIdx.y + blockDim.y * blockIdx.y;

    // TODO: remove for guaranteed inbounds
    if (xDst < destination.width() && yDst < destination.height()) {

        const int xSrc = 2 * xDst;
        const int ySrc = 2 * yDst;

        const Scalar center = source(xSrc,ySrc);

        ScalarComp totalWeight(0);
        ScalarComp totalValue(0);

        const int window = halfKernel.length();

        const int dxMin = max(-(window-1), -xSrc);
        const int dxMax = min( window, source.width() - xSrc - 1);

        const int dyMin = max(-(window-1), -ySrc);
        const int dyMax = min( window, source.height() - ySrc - 1);

        for (int dy = dyMin; dy <= dyMax; ++dy) {
            for (int dx = dxMin; dx <= dxMax; ++dx) {

                const Scalar sourceVal = source(xSrc + dx, ySrc + dy);

                const Scalar diff = sourceVal - center;

                if (diff < 3 * sigmaInput) {

                    const Scalar weight = halfKernel(dx > 0 ? (dx-1) : -dx) *
                                          halfKernel(dy > 0 ? (dy-1) : -dy) * expf( -(diff * diff) / (2 * sigmaInput * sigmaInput) );

                    totalWeight += weight;

                    totalValue += weight * sourceVal;

                }

            }
        }

        destination(xDst,yDst) = totalValue / totalWeight;

    }

}



template <typename Scalar, typename ScalarComp>
void radiallySymmetricBilateralFilterAndDownsampleBy2(DeviceTensor2<Scalar> & destination, const DeviceTensor2<Scalar> & source,
                                                      DeviceTensor1<ScalarComp> & halfKernel, const ScalarComp sigmaInput) {


    assert((source.width() / destination.width()) == 2);
    assert((source.height() / destination.height()) == 2);

    const dim3 block(16,16);
    const dim3 grid(intDivideAndCeil(destination.width(),block.x),
                    intDivideAndCeil(destination.height(),block.y));

    radiallySymmetricBilateralFilterAndDownsampleBy2Kernel<<<grid,block>>>(destination,source,halfKernel,sigmaInput);


}

template void radiallySymmetricBilateralFilterAndDownsampleBy2(DeviceTensor2<ushort> & destination, const DeviceTensor2<ushort> & source,
                                                               DeviceTensor1<float> & halfKernel, const float sigmaInput);

template void radiallySymmetricBilateralFilterAndDownsampleBy2(DeviceTensor2<float> & destination, const DeviceTensor2<float> & source,
                                                               DeviceTensor1<float> & halfKernel, const float sigmaInput);


} // namespace df
