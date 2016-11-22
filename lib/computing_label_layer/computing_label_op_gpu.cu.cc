#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "computing_label_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void ComputingLabel(const int nthreads, const Dtype* bottom_data,
    const Dtype* bottom_depth, const Dtype* bottom_meta_data, 
    const int height, const int width, const int num_meta_data,
    const int grid_size, const int num_classes, int* top_label) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w) coords in top label
    int n = index;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    Dtype depth = bottom_depth[index];

    // find the voxel for this pixel

    // backproject the pixel to 3D
    // format of the meta_data
    // projection matrix: meta_data[0 ~ 11]
    // camera center: meta_data[12, 13, 14]
    // voxel step size: meta_data[15, 16, 17]
    // voxel min value: meta_data[18, 19, 20]
    // backprojection matrix: meta_data[21 ~ 32]
    const Dtype* meta_data = bottom_meta_data + n * num_meta_data;
    int offset = 21;
    Dtype X = meta_data[offset + 0] * w + meta_data[offset + 1] * h + meta_data[offset + 2];
    Dtype Y = meta_data[offset + 3] * w + meta_data[offset + 4] * h + meta_data[offset + 5];
    Dtype Z = meta_data[offset + 6] * w + meta_data[offset + 7] * h + meta_data[offset + 8];
    Dtype W = meta_data[offset + 9] * w + meta_data[offset + 10] * h + meta_data[offset + 11];
    X /= W;
    Y /= W;
    Z /= W;

    // compute the ray
    Dtype RX = X - meta_data[12];
    Dtype RY = Y - meta_data[13];
    Dtype RZ = Z - meta_data[14];

    // compute the norm
    Dtype N = sqrt(RX*RX + RY*RY + RZ*RZ);
        
    // normalization
    RX /= N;
    RY /= N;
    RZ /= N;

    // compute the 3D points
    X = meta_data[12] + depth * RX;
    Y = meta_data[13] + depth * RY;
    Z = meta_data[14] + depth * RZ;

    // voxel location in 3D
    int vd = floor((X - meta_data[18]) / meta_data[15]);
    int vh = floor((Y - meta_data[19]) / meta_data[16]);
    int vw = floor((Z - meta_data[20]) / meta_data[17]);

    int label = 0;
    if (vd >= 0 && vd < grid_size && vh >= 0 && vh < grid_size && vw >= 0 && vw < grid_size)
    {
      Dtype maxval = -1;
      for (int c = 0; c < num_classes; c++)
      {
        Dtype val = bottom_data[(n * grid_size * grid_size * grid_size + vd * grid_size * grid_size + vh * grid_size + vw) * num_classes + c];
        if (val > maxval)
        {
          maxval = val;
          label = c;
        }
      }
    }
    top_label[index] = label;
  }
}

bool ComputingLabelLaucher(
    const float* bottom_data, const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int num_meta_data,
    const int grid_size, const int num_classes, int* top_label, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width;
  cudaError_t err;

  ComputingLabel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_depth, bottom_meta_data,
      height, width, num_meta_data, grid_size, num_classes, top_label);

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
