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
__global__ void ComputeFlowForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_weights, const Dtype* bottom_points, const Dtype* bottom_depth,
    const Dtype* bottom_meta_data, const int height, const int width, const int channels, const int num_meta_data,
    const int kernel_size, const float threshold, const float max_weight, Dtype* top_data, Dtype* top_weights, Dtype* top_points)
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
    top_weights[index] = 1;
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

      if (c == 0)
      {
        top_points[index_pixel * 3 + 0] = X;
        top_points[index_pixel * 3 + 1] = Y;
        top_points[index_pixel * 3 + 2] = Z;
      }

      // apply pose_live2world
      Dtype X1 = meta_data[30] * X + meta_data[31] * Y + meta_data[32] * Z + meta_data[33];
      Dtype Y1 = meta_data[34] * X + meta_data[35] * Y + meta_data[36] * Z + meta_data[37];
      Dtype Z1 = meta_data[38] * X + meta_data[39] * Y + meta_data[40] * Z + meta_data[41];

      // apply the intrinsic matrix
      Dtype x1 = meta_data[0] * X1 + meta_data[1] * Y1 + meta_data[2] * Z1;
      Dtype x2 = meta_data[3] * X1 + meta_data[4] * Y1 + meta_data[5] * Z1;
      Dtype x3 = meta_data[6] * X1 + meta_data[7] * Y1 + meta_data[8] * Z1;
      int px = round(x1 / x3);
      int py = round(x2 / x3);

      // averaging over a small neighborhood
      int count = 0;
      for (int x = px - kernel_size; x <= px + kernel_size; x++)
      {
        for (int y = py - kernel_size; y <= py + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            // assign data and weight
            int index_bottom = n * height * width + y * width + x;
            Dtype Z_prev = bottom_points[index_bottom * 3 + 2];
            if (fabs(Z_prev - Z1) < threshold)
            {
              top_data[index] = (bottom_data[index_bottom * channels + c] + count * top_data[index]) / (count + 1);
              Dtype weight = bottom_weights[index_bottom * channels + c];
              if (weight > max_weight)
                top_weights[index] = (max_weight + count * top_weights[index]) / (count + 1);
              else
                top_weights[index] = (weight + count * top_weights[index]) / (count + 1);
              count++;
            }
          }
        }
      }
    }
  }
}

// bottom_data: (batch_size, height, width, channels)
bool ComputeFlowForwardLaucher(
    const float* bottom_data, const float* bottom_weights, const float* bottom_points,
    const float* bottom_depth, const float* bottom_meta_data,
    const int batch_size, const int height, const int width,
    const int channels, const int num_meta_data, 
    const int kernel_size, const float threshold, const float max_weight,
    float* top_data, float* top_weights, float* top_points, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int output_size = batch_size * height * width * channels;
  ComputeFlowForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_weights, bottom_points, bottom_depth, bottom_meta_data, height, width, channels, num_meta_data,
      kernel_size, threshold, max_weight, top_data, top_weights, top_points);

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
__global__ void ComputeFlowBackward(const int nthreads, const Dtype* top_diff, const Dtype* top_diff_weights,
    const Dtype* bottom_weights, const Dtype* bottom_points, const float* bottom_depth, const Dtype* bottom_meta_data, 
    const Dtype* top_points, const int height, const int width, const int channels, const int num_meta_data, const int kernel_size,
    const float threshold, const float max_weight, Dtype* bottom_diff, Dtype* bottom_diff_weights) 
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
    {
      bottom_diff[index] = 0;
      bottom_diff_weights[index] = 0;
    }
    else
    {
      bottom_diff[index] = 0;
      bottom_diff_weights[index] = 0;
      // backproject the pixel to 3D
      // format of the meta_data
      // intrinsic matrix: meta_data[0 ~ 8]
      // inverse intrinsic matrix: meta_data[9 ~ 17]
      // pose_world2live: meta_data[18 ~ 29]
      // pose_live2world: meta_data[30 ~ 41]
      // voxel step size: meta_data[42, 43, 44]
      // voxel min value: meta_data[45, 46, 47]
      const Dtype* meta_data = bottom_meta_data + n * num_meta_data;

      // apply pose_world2live
      Dtype X1 = meta_data[18] * X_prev + meta_data[19] * Y_prev + meta_data[20] * Z_prev + meta_data[21];
      Dtype Y1 = meta_data[22] * X_prev + meta_data[23] * Y_prev + meta_data[24] * Z_prev + meta_data[25];
      Dtype Z1 = meta_data[26] * X_prev + meta_data[27] * Y_prev + meta_data[28] * Z_prev + meta_data[29];

      // apply the intrinsic matrix
      Dtype x1 = meta_data[0] * X1 + meta_data[1] * Y1 + meta_data[2] * Z1;
      Dtype x2 = meta_data[3] * X1 + meta_data[4] * Y1 + meta_data[5] * Z1;
      Dtype x3 = meta_data[6] * X1 + meta_data[7] * Y1 + meta_data[8] * Z1;
      int px = round(x1 / x3);
      int py = round(x2 / x3);

      // averaging over a small neighborhood
      int count = 0;
      for (int x = px - kernel_size; x <= px + kernel_size; x++)
      {
        for (int y = py - kernel_size; y <= py + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            // assign data
            int index_top = n * height * width + y * width + x;
            Dtype d = bottom_depth[index_top];
            if (fabs(d - Z1) < threshold)
            {
              bottom_diff[index] = (top_diff[index_top * channels + c] + count * bottom_diff[index]) / (count + 1);
              if (bottom_weights[index] > max_weight)
                bottom_diff_weights[index] = (0 + count * bottom_diff_weights[index]) / (count + 1);
              else
                bottom_diff_weights[index] = (top_diff_weights[index_top * channels + c] + count * bottom_diff_weights[index]) / (count + 1);
              count++;
            }
          }
        }
      }
    }
  }
}

 
bool ComputeFlowBackwardLaucher(const float* top_diff, const float* top_diff_weights, const float* bottom_weights, 
    const float* bottom_points, const float* bottom_depth, const float* bottom_meta_data, const float* top_points, const int batch_size,
    const int height, const int width, const int channels, const int num_meta_data, const int kernel_size, const float threshold, const float max_weight,
    float* bottom_diff, float* bottom_diff_weights, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  ComputeFlowBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, top_diff_weights, bottom_weights, bottom_points, bottom_depth, bottom_meta_data, top_points,
      height, width, channels, num_meta_data, kernel_size, threshold, max_weight, bottom_diff, bottom_diff_weights);

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
