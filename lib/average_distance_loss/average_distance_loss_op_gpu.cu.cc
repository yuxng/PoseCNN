#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <Eigen/Geometry>
#include "average_distance_loss_op_gpu.h"

#define POSE_CHANNELS 4

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename Dtype>
__global__ void AveragedistanceForward(const int nthreads, const Dtype* prediction, const Dtype* target,
    const Dtype* weight, const Dtype* point, const int batch_size, const int num_classes, 
    const int num_points, Dtype* rotations, Dtype* losses, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(n, nthreads) 
  {
    // find the class label and pose of this object
    int index_cls = -1, ind;
    Dtype s, u, v, w;
    for (int i = 0; i < POSE_CHANNELS * num_classes; i += POSE_CHANNELS)
    {
      int index = n * POSE_CHANNELS * num_classes + i;
      if (weight[index] > 0)
      {
        index_cls = i / POSE_CHANNELS;

        // gt quaternion
        s = target[index + 0];
        u = target[index + 1];
        v = target[index + 2];
        w = target[index + 3];

        // gt rotation matrix
        ind = n * 6 * 9;
        rotations[ind + 0] = s * s + u * u - v * v - w * w;
        rotations[ind + 1] = 2 * (u * v - s * w);
        rotations[ind + 2] = 2 * (u * w + s * v);
        rotations[ind + 3] = 2 * (u * v + s * w);
        rotations[ind + 4] = s * s - u * u + v * v - w * w;
        rotations[ind + 5] = 2 * (v * w - s * u);
        rotations[ind + 6] = 2 * (u * w - s * v);
        rotations[ind + 7] = 2 * (v * w + s * u);
        rotations[ind + 8] = s * s - u * u - v * v + w * w;

        // predicted quaternion
        s = prediction[index + 0];
        u = prediction[index + 1];
        v = prediction[index + 2];
        w = prediction[index + 3];

        // predicted rotation matrix
        ind = n * 6 * 9 + 9;
        rotations[ind + 0] = s * s + u * u - v * v - w * w;
        rotations[ind + 1] = 2 * (u * v - s * w);
        rotations[ind + 2] = 2 * (u * w + s * v);
        rotations[ind + 3] = 2 * (u * v + s * w);
        rotations[ind + 4] = s * s - u * u + v * v - w * w;
        rotations[ind + 5] = 2 * (v * w - s * u);
        rotations[ind + 6] = 2 * (u * w - s * v);
        rotations[ind + 7] = 2 * (v * w + s * u);
        rotations[ind + 8] = s * s - u * u - v * v + w * w;
        break;
      }
    }
    if (index_cls == -1)
      continue;

    // derivatives of Ru to quaternion
    ind = n * 6 * 9 + 18;
    rotations[ind + 0] = 2 * s;
    rotations[ind + 1] = -2 * w;
    rotations[ind + 2] = 2 * v;
    rotations[ind + 3] = 2 * w;
    rotations[ind + 4] = 2 * s;
    rotations[ind + 5] = -2 * u;
    rotations[ind + 6] = -2 * v;
    rotations[ind + 7] = 2 * u;
    rotations[ind + 8] = 2 * s;

    ind = n * 6 * 9 + 27;
    rotations[ind + 0] = 2 * u;
    rotations[ind + 1] = 2 * v;
    rotations[ind + 2] = 2 * w;
    rotations[ind + 3] = 2 * v;
    rotations[ind + 4] = -2 * u;
    rotations[ind + 5] = -2 * s;
    rotations[ind + 6] = 2 * w;
    rotations[ind + 7] = 2 * s;
    rotations[ind + 8] = -2 * u;

    ind = n * 6 * 9 + 36;
    rotations[ind + 0] = -2 * v;
    rotations[ind + 1] = 2 * u;
    rotations[ind + 2] = 2 * s;
    rotations[ind + 3] = 2 * u;
    rotations[ind + 4] = 2 * v;
    rotations[ind + 5] = 2 * w;
    rotations[ind + 6] = -2 * s;
    rotations[ind + 7] = 2 * w;
    rotations[ind + 8] = -2 * v;

    ind = n * 6 * 9 + 45;
    rotations[ind + 0] = -2 * w;
    rotations[ind + 1] = -2 * s;
    rotations[ind + 2] = 2 * u;
    rotations[ind + 3] = 2 * s;
    rotations[ind + 4] = -2 * w;
    rotations[ind + 5] = 2 * v;
    rotations[ind + 6] = 2 * u;
    rotations[ind + 7] = 2 * v;
    rotations[ind + 8] = 2 * w;

    // for each point
    Dtype diff0, diff1, diff2;
    for (int i = 0; i < num_points; i++)
    {
      int index = index_cls * num_points * 3 + i * 3;
      ind = n * 6 * 9;

      diff0 = rotations[ind + 9 + 0] * point[index + 0] + rotations[ind + 9 + 1] * point[index + 1] + rotations[ind + 9 + 2] * point[index + 2] 
              - rotations[ind + 0] * point[index + 0] - rotations[ind + 1] * point[index + 1] - rotations[ind + 2] * point[index + 2];

      diff1 = rotations[ind + 9 + 3] * point[index + 0] + rotations[ind + 9 + 4] * point[index + 1] + rotations[ind + 9 + 5] * point[index + 2] 
              - rotations[ind + 3] * point[index + 0] - rotations[ind + 4] * point[index + 1] - rotations[ind + 5] * point[index + 2];

      diff2 = rotations[ind + 9 + 6] * point[index + 0] + rotations[ind + 9 + 7] * point[index + 1] + rotations[ind + 9 + 8] * point[index + 2] 
              - rotations[ind + 6] * point[index + 0] - rotations[ind + 7] * point[index + 1] - rotations[ind + 8] * point[index + 2];

      losses[n] += (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) / (2.0 * batch_size * num_points);

      int index_diff = n * POSE_CHANNELS * num_classes + POSE_CHANNELS * index_cls;
      for (int j = 0; j < 3; j++)
      {
        Dtype diff;
        if (j == 0)
          diff = diff0;
        else if (j == 1)
          diff = diff1;
        else
          diff = diff2;
        for (int k = 0; k < 3; k++)
        {
          ind = n * 6 * 9 + 18;
          bottom_diff[index_diff + 0] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
          ind = n * 6 * 9 + 27;
          bottom_diff[index_diff + 1] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
          ind = n * 6 * 9 + 36;
          bottom_diff[index_diff + 2] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
          ind = n * 6 * 9 + 45;
          bottom_diff[index_diff + 3] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        }
      }
    }
  }
}

// bottom_data: (batch_size, 4 * num_classes)
void AveragedistanceForwardLaucher(OpKernelContext* context,
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const int batch_size, const int num_classes, const int num_points,
    float* top_data, float* bottom_diff, const Eigen::GpuDevice& d)
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size;

  // allocate temp memory
  int dim = batch_size;
  TensorShape output_shape_losses;
  TensorShapeUtils::MakeShape(&dim, 1, &output_shape_losses);
  Tensor losses_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_losses, &losses_tensor));
  float* losses = losses_tensor.flat<float>().data();
  checkCuda(cudaMemset(losses, 0, batch_size * sizeof(float)));

  dim = batch_size * 6 * 9;
  TensorShape output_shape_rotations;
  TensorShapeUtils::MakeShape(&dim, 1, &output_shape_rotations);
  Tensor rotations_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_rotations, &rotations_tensor));
  float* rotations = rotations_tensor.flat<float>().data();

  // compute the loss
  checkCuda(cudaMemset(top_data, 0, sizeof(float)));
  checkCuda(cudaMemset(bottom_diff, 0, batch_size * num_classes * POSE_CHANNELS * sizeof(float)));
  AveragedistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_prediction, bottom_target, bottom_weight, bottom_point, batch_size, num_classes, num_points, rotations, losses, bottom_diff);
  cudaDeviceSynchronize();

  // sum the loss and diffs
  thrust::device_ptr<float> losses_ptr(losses);
  float loss = thrust::reduce(losses_ptr, losses_ptr + output_size);
  cudaMemcpy(top_data, &loss, sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed hello: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }
}


template <typename Dtype>
__global__ void AveragedistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}

 
bool AveragedistanceBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int channels, float* output, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * channels;
  cudaError_t err;

  AveragedistanceBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, bottom_diff, output);

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
