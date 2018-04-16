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
    const Dtype* weight, const Dtype* point, const Dtype* symmetry, const int batch_size, const int num_classes, 
    const int num_points, const float margin, Dtype* rotations, Dtype* losses, Dtype* diffs) 
{
  CUDA_1D_KERNEL_LOOP(index_thread, nthreads) 
  {
    // batch index
    int n = index_thread / num_points;
    int p = index_thread % num_points;

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
        ind = n * num_points * 6 * 9 + p * 6 * 9;
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
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 9;
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
    ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
    rotations[ind + 0] = 2 * s;
    rotations[ind + 1] = -2 * w;
    rotations[ind + 2] = 2 * v;
    rotations[ind + 3] = 2 * w;
    rotations[ind + 4] = 2 * s;
    rotations[ind + 5] = -2 * u;
    rotations[ind + 6] = -2 * v;
    rotations[ind + 7] = 2 * u;
    rotations[ind + 8] = 2 * s;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
    rotations[ind + 0] = 2 * u;
    rotations[ind + 1] = 2 * v;
    rotations[ind + 2] = 2 * w;
    rotations[ind + 3] = 2 * v;
    rotations[ind + 4] = -2 * u;
    rotations[ind + 5] = -2 * s;
    rotations[ind + 6] = 2 * w;
    rotations[ind + 7] = 2 * s;
    rotations[ind + 8] = -2 * u;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
    rotations[ind + 0] = -2 * v;
    rotations[ind + 1] = 2 * u;
    rotations[ind + 2] = 2 * s;
    rotations[ind + 3] = 2 * u;
    rotations[ind + 4] = 2 * v;
    rotations[ind + 5] = 2 * w;
    rotations[ind + 6] = -2 * s;
    rotations[ind + 7] = 2 * w;
    rotations[ind + 8] = -2 * v;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
    rotations[ind + 0] = -2 * w;
    rotations[ind + 1] = -2 * s;
    rotations[ind + 2] = 2 * u;
    rotations[ind + 3] = 2 * s;
    rotations[ind + 4] = -2 * w;
    rotations[ind + 5] = 2 * v;
    rotations[ind + 6] = 2 * u;
    rotations[ind + 7] = 2 * v;
    rotations[ind + 8] = 2 * w;

    // for the point
    int index = index_cls * num_points * 3 + p * 3;
    ind = n * num_points * 6 * 9 + p * 6 * 9;

    // rotate the first point
    Dtype x1 = rotations[ind + 9 + 0] * point[index + 0] + rotations[ind + 9 + 1] * point[index + 1] + rotations[ind + 9 + 2] * point[index + 2];
    Dtype y1 = rotations[ind + 9 + 3] * point[index + 0] + rotations[ind + 9 + 4] * point[index + 1] + rotations[ind + 9 + 5] * point[index + 2];
    Dtype z1 = rotations[ind + 9 + 6] * point[index + 0] + rotations[ind + 9 + 7] * point[index + 1] + rotations[ind + 9 + 8] * point[index + 2];

    int index_min;
    Dtype x2, y2, z2;
    if (symmetry[index_cls] > 0)
    {
      // find the closet point for symmetry object
      Dtype dmin = FLT_MAX;
      for (int i = 0; i < num_points; i++)
      {
        int index2 = index_cls * num_points * 3 + i * 3;
        x2 = rotations[ind + 0] * point[index2 + 0] + rotations[ind + 1] * point[index2 + 1] + rotations[ind + 2] * point[index2 + 2];
        y2 = rotations[ind + 3] * point[index2 + 0] + rotations[ind + 4] * point[index2 + 1] + rotations[ind + 5] * point[index2 + 2];
        z2 = rotations[ind + 6] * point[index2 + 0] + rotations[ind + 7] * point[index2 + 1] + rotations[ind + 8] * point[index2 + 2];
        Dtype distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        if (distance < dmin)
        {
          dmin = distance;
          index_min = index2;
        }
      }
    }
    else
      index_min = index;

    x2 = rotations[ind + 0] * point[index_min + 0] + rotations[ind + 1] * point[index_min + 1] + rotations[ind + 2] * point[index_min + 2];
    y2 = rotations[ind + 3] * point[index_min + 0] + rotations[ind + 4] * point[index_min + 1] + rotations[ind + 5] * point[index_min + 2];
    z2 = rotations[ind + 6] * point[index_min + 0] + rotations[ind + 7] * point[index_min + 1] + rotations[ind + 8] * point[index_min + 2];

    Dtype distance = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    if (distance < margin)
      continue;

    losses[index_thread] = (distance - margin) / (2.0 * batch_size * num_points);

    int index_diff = n * num_points * POSE_CHANNELS * num_classes + p * POSE_CHANNELS * num_classes + POSE_CHANNELS * index_cls;
    for (int j = 0; j < 3; j++)
    {
      Dtype diff;
      if (j == 0)
        diff = x1 - x2;
      else if (j == 1)
        diff = y1 - y2;
      else
        diff = z1 - z2;
      for (int k = 0; k < 3; k++)
      {
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
        diffs[index_diff + 0] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
        diffs[index_diff + 1] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
        diffs[index_diff + 2] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
        diffs[index_diff + 3] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
      }
    }
  }
}


template <typename Dtype>
__global__ void sum_losses_gradients(const int nthreads, const Dtype* losses, const Dtype* diffs, const int batch_size, 
    const int num_classes, const int num_points, Dtype* loss_batch, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int n = index / (POSE_CHANNELS * num_classes);
    int c = index % (POSE_CHANNELS * num_classes);
/*
    // find the most violated point
    Dtype lmax = -FLT_MAX;
    int pmax;
    for (int p = 0; p < num_points; p++)
    {
      if (losses[n * num_points + p] > lmax)
      {
        lmax = losses[n * num_points + p];
        pmax = p;
      }
    }

    int index_diff = n * num_points * POSE_CHANNELS * num_classes + pmax * POSE_CHANNELS * num_classes + c;
    bottom_diff[index] = diffs[index_diff] * num_points;

    if (c == 0)
      loss_batch[n] = lmax * num_points;
*/    

    bottom_diff[index] = 0;
    for (int p = 0; p < num_points; p++)
    {
      int index_diff = n * num_points * POSE_CHANNELS * num_classes + p * POSE_CHANNELS * num_classes + c;
      bottom_diff[index] += diffs[index_diff];
    }

    if (c == 0)
    {
      loss_batch[n] = 0;
      for (int p = 0; p < num_points; p++)
        loss_batch[n] += losses[n * num_points + p];
    }

  }
}


// bottom_data: (batch_size, 4 * num_classes)
void AveragedistanceForwardLaucher(OpKernelContext* context,
    const float* bottom_prediction, const float* bottom_target, const float* bottom_weight, const float* bottom_point,
    const float* bottom_symmetry, const int batch_size, const int num_classes, const int num_points, const float margin,
    float* top_data, float* bottom_diff, const Eigen::GpuDevice& d)
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 1024;
  int output_size;

  // temp losses
  int dims[2];
  dims[0] = batch_size;
  dims[1] = num_points;
  TensorShape output_shape_losses;
  TensorShapeUtils::MakeShape(dims, 2, &output_shape_losses);
  Tensor losses_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_losses, &losses_tensor));
  float* losses = losses_tensor.flat<float>().data();
  checkCuda(cudaMemset(losses, 0, batch_size * num_points * sizeof(float)));

  TensorShape output_shape_loss_batch;
  TensorShapeUtils::MakeShape(&batch_size, 1, &output_shape_loss_batch);
  Tensor loss_batch_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_loss_batch, &loss_batch_tensor));
  float* loss_batch = loss_batch_tensor.flat<float>().data();
  checkCuda(cudaMemset(loss_batch, 0, batch_size * sizeof(float)));

  // temp diffs
  int dims_diff[3];
  dims_diff[0] = batch_size;
  dims_diff[1] = num_points;
  dims_diff[2] = POSE_CHANNELS * num_classes;
  TensorShape output_shape_diff;
  TensorShapeUtils::MakeShape(dims_diff, 3, &output_shape_diff);
  Tensor diffs_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_diff, &diffs_tensor));
  float* diffs = diffs_tensor.flat<float>().data();
  checkCuda(cudaMemset(diffs, 0, batch_size * num_points * POSE_CHANNELS * num_classes * sizeof(float)));

  // temp rotations
  int dims_rot[3];
  dims_rot[0] = batch_size;
  dims_rot[1] = num_points;
  dims_rot[2] = 6 * 9;
  TensorShape output_shape_rotations;
  TensorShapeUtils::MakeShape(dims_rot, 3, &output_shape_rotations);
  Tensor rotations_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_rotations, &rotations_tensor));
  float* rotations = rotations_tensor.flat<float>().data();
  checkCuda(cudaMemset(rotations, 0, batch_size * num_points * 6 * 9 * sizeof(float)));

  // compute the losses and gradients
  output_size = batch_size * num_points;
  AveragedistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_prediction, bottom_target, bottom_weight, bottom_point, bottom_symmetry,
      batch_size, num_classes, num_points, margin, rotations, losses, diffs);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the diffs
  checkCuda(cudaMemset(bottom_diff, 0, batch_size * POSE_CHANNELS * num_classes * sizeof(float)));
  output_size = batch_size * POSE_CHANNELS * num_classes;
  sum_losses_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, losses, diffs, batch_size, num_classes, num_points, loss_batch, bottom_diff);
  cudaDeviceSynchronize();

  // sum the loss
  checkCuda(cudaMemset(top_data, 0, sizeof(float)));
  thrust::device_ptr<float> losses_ptr(loss_batch);
  float loss = thrust::reduce(losses_ptr, losses_ptr + batch_size);
  cudaMemcpy(top_data, &loss, sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
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
