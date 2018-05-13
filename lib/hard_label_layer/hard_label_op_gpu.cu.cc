#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "hard_label_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void HardlabelForward(const int nthreads, const float* bottom_prob, const int* bottom_gt, 
  const int num_classes, const float threshold, Dtype* top_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    for (int c = 0; c < num_classes; c++)
      top_data[index * num_classes + c] = 0.0;

    int gt_label = bottom_gt[index];
    if (gt_label != -1 && (gt_label > 0 || bottom_prob[index * num_classes + gt_label] < threshold))
      top_data[index * num_classes + gt_label] = 1.0;
  }
}


bool HardlabelForwardLaucher(const float* bottom_prob, const int* bottom_gt,
  const int batch_size, const int height, const int width, const int num_classes,
  const float threshold, float* top_data, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width;
  cudaError_t err;

  HardlabelForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(output_size, bottom_prob, bottom_gt, num_classes, threshold, top_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void HardlabelBackward(const int nthreads, const int num_classes, Dtype* bottom_diff_prob, Dtype* bottom_diff_gt) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    bottom_diff_gt[index] = 0;
    for (int c = 0; c < num_classes; c++)
      bottom_diff_prob[index * num_classes + c] = 0;
  }
}


bool HardlabelBackwardLaucher(const float* top_diff, const int batch_size, const int height, const int width, const int num_classes,
    float* bottom_diff_prob, float* bottom_diff_gt, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width;
  cudaError_t err;

  HardlabelBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, num_classes, bottom_diff_prob, bottom_diff_gt);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
