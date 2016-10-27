#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "backprojecting_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void BackprojectForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_data_3d, const Dtype* bottom_depth,
    const Dtype* bottom_meta_data, const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, const float threshold, Dtype* top_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, d, h, w, c) is an element in the output
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % grid_size;
    n /= grid_size;
    int h = n % grid_size;
    n /= grid_size;
    int d = n % grid_size;
    n /= grid_size;

    // voxel location in 3D
    const Dtype* meta_data = bottom_meta_data + n * num_meta_data;
    Dtype X = d * meta_data[15] + meta_data[18];
    Dtype Y = h * meta_data[16] + meta_data[19];
    Dtype Z = w * meta_data[17] + meta_data[20];

    // project the 3D point to image
    Dtype x1 = meta_data[0] * X + meta_data[1] * Y + meta_data[2] * Z + meta_data[3];
    Dtype x2 = meta_data[4] * X + meta_data[5] * Y + meta_data[6] * Z + meta_data[7];
    Dtype x3 = meta_data[8] * X + meta_data[9] * Y + meta_data[10] * Z + meta_data[11];
    int px = round(x1 / x3);
    int py = round(x2 / x3);

    int flag = 0;
    if (px >= 0 && px < width && py >= 0 && py < height)
    {
      int index_pixel = n * height * width + py * width + px;
      Dtype depth = bottom_depth[index_pixel];

      // distance of this voxel to camera center
      Dtype dvoxel = sqrt((X - meta_data[12]) * (X - meta_data[12]) 
                        + (Y - meta_data[13]) * (Y - meta_data[13]) 
                        + (Z - meta_data[14]) * (Z - meta_data[14]));

      // check if the voxel is on the surface
      if (fabs(depth - dvoxel) < threshold)
      {
        flag = 1;
        top_data[index] = bottom_data[index_pixel * channels + c];
      }
    }

    if (flag == 0)
    {
      top_data[index] = bottom_data_3d[index]; 
    }

  }
}

// bottom_data: (batch_size, height, width, channels)
bool BackprojectForwardLaucher(
    const float* bottom_data, const float* bottom_data_3d,
    const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width, const int channels, const int num_meta_data, 
    const int grid_size, const float threshold,
    float* top_data, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int output_size = batch_size * grid_size * grid_size * grid_size * channels;
  BackprojectForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_data_3d, bottom_depth, bottom_meta_data, height, width, channels, num_meta_data,
      grid_size, threshold, top_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void BackprojectBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_depth, const Dtype* bottom_meta_data, 
    const int height, const int width, const int channels, const int num_meta_data,
    const int grid_size, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) coords in bottom data
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    int index_pixel = n * height * width + h * width + w;
    Dtype depth = bottom_depth[index_pixel];

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

    // get the gradient
    if (vd >= 0 && vd < grid_size && vh >= 0 && vh < grid_size && vw >= 0 && vw < grid_size)
      bottom_diff[index] = top_diff[(n * grid_size * grid_size * grid_size + vd * grid_size * grid_size + vh * grid_size + vw) * channels + c];
    else
      bottom_diff[index] = 0;
  }
}


bool BackprojectBackwardLaucher(const float* top_diff, const float* bottom_depth, const float* bottom_meta_data, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int grid_size, 
    float* bottom_diff, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  BackprojectBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, bottom_depth, bottom_meta_data,
      height, width, channels, num_meta_data, grid_size, bottom_diff);

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
