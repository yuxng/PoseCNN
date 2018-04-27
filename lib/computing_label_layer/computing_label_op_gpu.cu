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
    // intrinsic matrix: meta_data[0 ~ 8]
    // inverse intrinsic matrix: meta_data[9 ~ 17]
    // pose_world2live: meta_data[18 ~ 29]
    // pose_live2world: meta_data[30 ~ 41]
    // voxel step size: meta_data[42, 43, 44]
    // voxel min value: meta_data[45, 46, 47]
    const Dtype* meta_data = bottom_meta_data + n * num_meta_data;
    // apply the inverse intrinsic matrix
    int offset = 9;
    Dtype RX = meta_data[offset + 0] * w + meta_data[offset + 1] * h + meta_data[offset + 2];
    Dtype RY = meta_data[offset + 3] * w + meta_data[offset + 4] * h + meta_data[offset + 5];
    Dtype RZ = meta_data[offset + 6] * w + meta_data[offset + 7] * h + meta_data[offset + 8];

    // compute the 3D points in camera's coordinate system
    Dtype X = depth * RX;
    Dtype Y = depth * RY;
    Dtype Z = depth * RZ;

    // apply pose_live2world
    Dtype X1 = meta_data[30] * X + meta_data[31] * Y + meta_data[32] * Z + meta_data[33];
    Dtype Y1 = meta_data[34] * X + meta_data[35] * Y + meta_data[36] * Z + meta_data[37];
    Dtype Z1 = meta_data[38] * X + meta_data[39] * Y + meta_data[40] * Z + meta_data[41];

    // voxel location in 3D
    int vd = round((X1 - meta_data[45]) / meta_data[42]);
    int vh = round((Y1 - meta_data[46]) / meta_data[43]);
    int vw = round((Z1 - meta_data[47]) / meta_data[44]);

    int label = 0;
    if (vd >= 0 && vd < grid_size && vh >= 0 && vh < grid_size && vw >= 0 && vw < grid_size)
    {
      Dtype maxval;
      for (int c = 0; c < num_classes; c++)
      {
        Dtype val = bottom_data[(n * grid_size * grid_size * grid_size + vd * grid_size * grid_size + vh * grid_size + vw) * num_classes + c];
        if (c == 0)
        {
          maxval = val;
          label = c;
        }
        else
        {
          if (val > maxval)
          {
            maxval = val;
            label = c;
          }
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
