#include <df/surface/decimation.h>

#include <df/util/cudaHelpers.h>

namespace df {

namespace internal {

static constexpr char decimationStatusCovered = 1;
static constexpr char decimationStatusSelected = 2;

} // namespace internal

template <typename Scalar>
__global__ void decimateKernel(Tensor<3,Scalar,DeviceResident> vertices,
                               Tensor<3,Scalar,DeviceResident> decimatedVertices,
                               const Scalar radius,
                               Tensor<1,bool,DeviceResident> covered,
                               const int blockSize,
                               int * nVertices) {

    const uint vSource = threadIdx.x + blockDim.x * blockIdx.x;
    const uint targetBlock = threadIdx.y + blockDim.y * blockIdx.y;

    const uint numVertices = vertices.dimensionSize(1);

    if (vSource < numVertices) {

        if (!covered(vSource)) {

            const uint rangeBegin = targetBlock*blockSize;
            const uint rangeEnd = min((targetBlock+1)*blockSize,numVertices);

//            for (uint vTarget = rangeBegin; )

        }

    }

}

template <typename Scalar>
void decimate(Tensor<3,Scalar,DeviceResident> & vertices,
              Tensor<3,Scalar,DeviceResident> & decimatedVertices,
              const Scalar radius) {

//    const uint numVertices = vertices.dimensionSize(1);

//    ManagedTensor<1,bool,DeviceResident> covered(Eigen::Matrix<uint,1,1>(numVertices));
//    cudaMemset(covered.data(),0,covered.count()*sizeof(bool));

//    ManagedTensor<1,int,DeviceResident> vertexCount(Eigen::Matrix<uint,1,1>(1));
//    cudaMemset(vertexCount.data(),0,sizeof(int));

//    const uint blockSize = 1024;

//    const dim3 block(1024,1,1);
//    const dim3 grid(intDivideAndCeil(numVertices,block.x),intDivideAndCeil(numVertices,blockSize),1);

//    decimateKernel<<<grid,block>>>(vertices,decimatedVertices,radius,covered,blockSize,numVertices.data());

//    cudaDeviceSynchronize();
//    CheckCudaDieOnError();

}

//template void decimate(Tensor<3,float,DeviceResident> & vertices,
//                       Tensor<3,float,DeviceResident> & decimatedVertices);

} // namespace df
