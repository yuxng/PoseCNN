#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <math.h>
#include "computing_flow_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void ComputeFlowForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_points, const Dtype* bottom_depth,
    const Dtype* bottom_meta_data, const int height, const int width, const int channels, const int num_meta_data,
    const int kernel_size, const float threshold, Dtype* top_data, Dtype* top_points)
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

    // initialization
    top_data[index] = 0;
    if (c == 0)
    {
      top_points[index_pixel * 3 + 0] = NAN;
      top_points[index_pixel * 3 + 1] = NAN;
      top_points[index_pixel * 3 + 2] = NAN;
    }

    Dtype depth = bottom_depth[index_pixel];
    if (depth > 0)
    {
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

      if (c == 0)
      {
        top_points[index_pixel * 3 + 0] = X1;
        top_points[index_pixel * 3 + 1] = Y1;
        top_points[index_pixel * 3 + 2] = Z1;
      }

      // check a neighborhood around (w, h)
      Dtype dmin = 1000.0;
      int fx = -1;
      int fy = -1;
      for (int x = w - kernel_size; x <= w + kernel_size; x++)
      {
        for (int y = h - kernel_size; y <= h + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            int index_bottom = n * height * width + y * width + x;
            Dtype X_prev = bottom_points[index_bottom * 3 + 0];
            Dtype Y_prev = bottom_points[index_bottom * 3 + 1];
            Dtype Z_prev = bottom_points[index_bottom * 3 + 2];
            if (isnan(X_prev) || isnan(Y_prev) || isnan(Z_prev))
              continue;

            // distance
            Dtype dis = sqrt((X1 - X_prev) * (X1 - X_prev) + (Y1 - Y_prev) * (Y1 - Y_prev) + (Z1 - Z_prev) * (Z1 - Z_prev));
            if (dis < dmin)
            {
              dmin = dis;
              fx = x;
              fy = y;
            }
          }
        }
      }

      if (dmin < threshold)
      {
        // assign data
        int index_bottom = n * height * width + fy * width + fx;
        top_data[index] = bottom_data[index_bottom * channels + c];
      }
    }
  }
}

// bottom_data: (batch_size, height, width, channels)
bool ComputeFlowForwardLaucher(
    const float* bottom_data, const float* bottom_points,
    const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width,
    const int channels, const int num_meta_data, 
    const int kernel_size, const float threshold,
    float* top_data, float* top_points, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int output_size = batch_size * height * width * channels;
  ComputeFlowForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_points, bottom_depth, bottom_meta_data, height, width, channels, num_meta_data,
      kernel_size, threshold, top_data, top_points);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void ComputeFlowBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_points, const Dtype* top_points, 
    const int height, const int width, const int channels, const int kernel_size,
    const float threshold, Dtype* bottom_diff) 
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
    Dtype X_prev = bottom_points[index_pixel * 3 + 0];
    Dtype Y_prev = bottom_points[index_pixel * 3 + 1];
    Dtype Z_prev = bottom_points[index_pixel * 3 + 2];
    if (isnan(X_prev) || isnan(Y_prev) || isnan(Z_prev))
      bottom_diff[index] = 0;
    else
    {
      // check a neighborhood around (w, h)
      Dtype dmin = 1000.0;
      int fx = -1;
      int fy = -1;
      for (int x = w - kernel_size; x <= w + kernel_size; x++)
      {
        for (int y = h - kernel_size; y <= h + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            int index_top = n * height * width + y * width + x;
            Dtype X1 = top_points[index_top * 3 + 0];
            Dtype Y1 = top_points[index_top * 3 + 1];
            Dtype Z1 = top_points[index_top * 3 + 2];
            if (isnan(X1) || isnan(Y1) || isnan(Z1))
              continue;

            // distance
            Dtype dis = sqrt((X1 - X_prev) * (X1 - X_prev) + (Y1 - Y_prev) * (Y1 - Y_prev) + (Z1 - Z_prev) * (Z1 - Z_prev));
            if (dis < dmin)
            {
              dmin = dis;
              fx = x;
              fy = y;
            }
          }
        }
      }

      if (dmin < threshold)
      {
        // assign data
        int index_top = n * height * width + fy * width + fx;
        bottom_diff[index] = top_diff[index_top * channels + c];
      }
      else
        bottom_diff[index] = 0;
    }
  }
}

 
bool ComputeFlowBackwardLaucher(const float* top_diff, const float* bottom_points, const float* top_points, 
    const int batch_size, const int height, const int width, const int channels, const int kernel_size, const float threshold,
    float* bottom_diff, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  ComputeFlowBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, bottom_points, top_points,
      height, width, channels, kernel_size, threshold, bottom_diff);

  cudaDeviceSynchronize();
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
