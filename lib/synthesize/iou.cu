#include <stdio.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <df/util/cudaHelpers.h>
#include <df/util/macros.h>
#include <df/util/eigenHelpers.h>
#include <df/util/tensor.h>

namespace df {

__global__ void iouKernel(const DeviceTensor2<int> labelMap, DeviceTensor2<int> interMap, DeviceTensor2<int> unionMap,
                                DeviceTensor2<Eigen::UnalignedVec4<float> > vertMap, int classID)
{
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

  float vx = vertMap(x, y)(0);
  int label = std::round(vx) + 1;
  int label_pred = labelMap(x, y);

  if (label == classID || label_pred == classID)
    unionMap(x, y) = 1;
  else
    unionMap(x, y) = 0;

  if (label == classID && label_pred == classID)
    interMap(x, y) = 1;
  else
    interMap(x, y) = 0;
}

float iou(const DeviceTensor2<int> & labelMap, DeviceTensor2<int> & interMap, DeviceTensor2<int> & unionMap,
                DeviceTensor2<Eigen::UnalignedVec4<float> > & vertMap, int classID)
{
  const dim3 block(16,16,1);
  const dim3 grid(intDivideAndCeil(labelMap.dimensionSize(0), block.x),
                  intDivideAndCeil(labelMap.dimensionSize(1), block.y),
                  1);
  iouKernel<<<grid,block>>>(labelMap, interMap, unionMap, vertMap, classID);

  // sum the loss and diffs
  thrust::device_ptr<int> inter_ptr(interMap.data());
  float inter_value = thrust::reduce(inter_ptr, inter_ptr + interMap.count());

  thrust::device_ptr<int> union_ptr(unionMap.data());
  float union_value = thrust::reduce(union_ptr, union_ptr + unionMap.count());

  cudaDeviceSynchronize();
  CheckCudaDieOnError();

  return inter_value / union_value;
}


}
