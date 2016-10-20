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

__global__ void Initialize(const int nthreads, int* top_voxel_locations) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    top_voxel_locations[index] = -1;
  }
}

template <typename Dtype>
__global__ void BackprojectForward(const int nthreads, const Dtype* bottom_data,
    const int* bottom_pixel_locations, const int height, const int width, const int channels,
    const int grid_size, const int channels_location,
    Dtype* top_data, int* top_count, int* top_voxel_locations) 
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

    // loop over the pixels in this voxel
    Dtype val = 0;
    int count = 0;
    int offset_location = (n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w) * channels_location;
    for(int c_location = 0; c_location < channels_location; c_location++)
    {
      int location = bottom_pixel_locations[offset_location + c_location];
      if (location > 0)
      {
        count++;
        int pixel_w = location % width;
        int pixel_h = location / width;
        int index_pixel = (n * height * width + pixel_h * width + pixel_w) * channels + c;
        val += bottom_data[index_pixel];

        // store the voxel location
        top_voxel_locations[n * height * width + pixel_h * width + pixel_w] = d * grid_size * grid_size + h * grid_size + w;
      }
    }
    // compute the mean
    if (count > 1)
      val /= count;
    top_data[index] = val;
    // store count
    if (c == 0)
      top_count[n * grid_size * grid_size * grid_size + d * grid_size * grid_size + h * grid_size + w] = count;
  }
}

// bottom_data: (batch_size, height, width, channels)
// bottom_pixel_locations: (batch_size, grid_size, grid_size, grid_size, channels_location)
bool BackprojectForwardLaucher(
    const float* bottom_data, const int* bottom_pixel_locations,
    const int batch_size, const int height, const int width, const int channels,
    const int grid_size, const int channels_location,
    float* top_data, int* top_count, int* top_voxel_locations, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  // initialize the top_voxel_locations
  const int output_size_voxel = batch_size * height * width;
  Initialize<<<(output_size_voxel + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(output_size_voxel, top_voxel_locations);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  const int output_size = batch_size * grid_size * grid_size * grid_size * channels;
  BackprojectForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_pixel_locations, height, width, channels,
      grid_size, channels_location, top_data, top_count, top_voxel_locations);

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
    const int* top_count, const int* top_voxel_locations, 
    const int height, const int width, const int channels, 
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

    Dtype gradient = 0;
    // find the voxel for this pixel
    int location = top_voxel_locations[n * height * width + h * width + w];
    if (location > 0)
    {
      int voxel_w = location % grid_size;
      location /= grid_size;
      int voxel_h = location % grid_size;
      int voxel_d = location / grid_size;

      int index_top_base = n * grid_size * grid_size * grid_size + voxel_d * grid_size * grid_size + voxel_h * grid_size + voxel_w;
      int index_top = index_top_base * channels + c;
      gradient = top_diff[index_top];
      int count = top_count[index_top_base];
      if (count > 1)
        gradient /= count;
    }
    bottom_diff[index] = gradient;
  }
}


bool BackprojectBackwardLaucher(const float* top_diff, const int* top_count, const int* top_voxel_locations, 
    const int batch_size, const int height, const int width, const int channels, const int grid_size,
    float* bottom_diff, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  BackprojectBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, top_count, top_voxel_locations, 
      height, width, channels, grid_size, bottom_diff);

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
