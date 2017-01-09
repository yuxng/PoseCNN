#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>
#include "triplet_loss_op_gpu.h"

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

__global__ void init_state(const int nthreads, unsigned int seed, curandState_t* states) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    curand_init(seed, index, 0, &states[index]);
  }
}

template <typename Dtype>
__global__ void TripletForward(const int nthreads, const Dtype* bottom_data, int* triplets,
    const int channels, const float margin, Dtype* losses, Dtype* diffs) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // compute the distances
    int index_i = triplets[index * 3 + 0];
    int index_j = triplets[index * 3 + 1];
    int index_k = triplets[index * 3 + 2];

    Dtype D_ij = 0;
    Dtype D_ik = 0;
    for (int c = 0; c < channels; c++)
    {
      D_ij += (bottom_data[index_i * channels + c] - bottom_data[index_j * channels + c]) * (bottom_data[index_i * channels + c] - bottom_data[index_j * channels + c]);
      D_ik += (bottom_data[index_i * channels + c] - bottom_data[index_k * channels + c]) * (bottom_data[index_i * channels + c] - bottom_data[index_k * channels + c]);
    }

    // store the loss
    Dtype dis = D_ij - D_ik + margin;
    losses[index] = max(dis, Dtype(0.0));

    // compute gradients
    if (dis > 0)
    {
      for (int c = 0; c < channels; c++)
      {
        // update x_i
        diffs[index * channels * 3 + c] = (bottom_data[index_k * channels + c] - bottom_data[index_j * channels + c]) / nthreads;
        // update x_j
        diffs[index * channels * 3 + channels + c] = (bottom_data[index_j * channels + c] - bottom_data[index_i * channels + c]) / nthreads;
        // update x_k
        diffs[index * channels * 3 + 2 * channels + c] = (bottom_data[index_i * channels + c] - bottom_data[index_k * channels + c]) / nthreads;
      }
    }
  }
}

template <typename Dtype>
__global__ void sum_gradients(const int nthreads, const Dtype* diffs, const int* triplets,
    const int batch_size, const int height, const int width, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int c = index;
    int channels = nthreads;
    // for each triplet
    for (int n = 0; n < batch_size; n++)
    {
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w++)
        {
          int index_triplet = n * height * width + h * width + w;
          int index_i = triplets[index_triplet * 3 + 0];
          int index_j = triplets[index_triplet * 3 + 1];
          int index_k = triplets[index_triplet * 3 + 2];

          bottom_diff[index_i * channels + c] += diffs[index_triplet * channels * 3 + c];
          bottom_diff[index_j * channels + c] += diffs[index_triplet * channels * 3 + channels + c];
          bottom_diff[index_k * channels + c] += diffs[index_triplet * channels * 3 + 2 * channels + c];
        }
      }
    }
  }
}

// bottom_data: (batch_size, height, width, channels)
bool TripletForwardLaucher(
    const float* bottom_data, const float* bottom_label, const int* bottom_prediction,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d)
{
  // copy labels to CPU
  float* labels = (float*)malloc(batch_size * height * width * num_classes * sizeof(float));
  cudaMemcpy(labels, bottom_label, batch_size * height * width * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

  // copy predictions to CPU
  int* predictions = (int*)malloc(batch_size * height * width * sizeof(int));
  cudaMemcpy(predictions, bottom_prediction, batch_size * height * width * sizeof(int), cudaMemcpyDeviceToHost);

  // sample triplets to define the loss
  // compute label indexes
  std::vector< std::vector<int> > label_indexes(num_classes);
  std::vector< std::vector<int> > label_indexes_correct(num_classes);
  for (int n = 0; n < batch_size; n++)
  {
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        int index = n * height * width + h * width + w;
        int cls;
        for (int c = 0; c < num_classes; c++)
        {
          if(labels[index * num_classes + c] > 0)
          {
            label_indexes[c].push_back(index);
            cls = c;
            break;
          }
        }
        if (predictions[index] == cls)
          label_indexes_correct[cls].push_back(index);
      } 
    }
  }

  // classes in the batch
  std::vector<int> class_indexes;
  for (int i = 0; i < num_classes; i++)
  {
    if (label_indexes[i].size() > 0)
    {
      class_indexes.push_back(i);
    }
  }

  // sampling
  std::srand ( unsigned ( std::time(0) ) );
  std::vector<int> triplets(batch_size * height * width * 3);
  for (int n = 0; n < batch_size; n++)
  {
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        // anchor
        int index = n * height * width + h * width + w;
        int cls;
        for (int c = 0; c < num_classes; c++)
        {
          if(labels[index * num_classes + c] > 0)
          {
            cls = c;
            break;
          }
        }

        // sample a positive pixel
        int num = label_indexes_correct[cls].size();
        int index_p;
        if (num > 0 && rand() % 2 == 0)
        {
          if (num == 1)
            index_p = label_indexes_correct[cls][0];
          else
          {
            while(1)
            {
              index_p = label_indexes_correct[cls][rand() % num];
              if (index_p != index)
                break;
            }
          }
        }
        else
        {
          num = label_indexes[cls].size();
          if (num == 1)
            index_p = index;
          else
          {
            while(1)
            {
              index_p = label_indexes[cls][rand() % num];
              if (index_p != index)
                break;
            }
          }
        }

        // sample a negative pixel
        int cls_neg;
        // check the predicted label of this pixel for hard negative
        int cls_pred = predictions[index];
        if (cls_pred != cls && label_indexes[cls_pred].size() > 0 && rand() % 2 == 0)
          cls_neg = cls_pred;
        else
        {
          while(1)
          {
            cls_neg = class_indexes[rand() % class_indexes.size()];
            if (cls_neg != cls)
              break;
          }
        }
        int index_n;
        num = label_indexes_correct[cls_neg].size();
        if (num > 0 && rand() % 2 == 0)
          index_n = label_indexes_correct[cls_neg][rand() % num];
        else
        {
          num = label_indexes[cls_neg].size();
          index_n = label_indexes[cls_neg][rand() % num];
        }

        // store the triplet
        triplets[index * 3 + 0] = index;
        triplets[index * 3 + 1] = index_p;
        triplets[index * 3 + 2] = index_n;
      } 
    }
  }

  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width;

  // compute the loss matrix
  float* losses;
  float* diffs;
  int* triplets_device;
  checkCuda(cudaMalloc((void **) &losses, output_size * sizeof(float))); 
  checkCuda(cudaMalloc((void **) &diffs, output_size * channels * 3 * sizeof(float))); 
  checkCuda(cudaMalloc((void **) &triplets_device, output_size * 3 * sizeof(int)));
  cudaMemcpy(triplets_device, triplets.data(), output_size * 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(diffs, 0, output_size * channels * 3 * sizeof(float));

  TripletForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, triplets_device, channels, margin, losses, diffs);
  cudaDeviceSynchronize();
  
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the loss and diffs
  thrust::device_ptr<float> losses_ptr(losses);
  float loss = thrust::reduce(losses_ptr, losses_ptr + output_size);
  loss /= output_size * 2.0;
  cudaMemcpy(top_data, &loss, sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(bottom_diff, 0, batch_size * height * width * channels * sizeof(float));
  sum_gradients<<<(channels + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      channels, diffs, triplets_device, batch_size, height, width, bottom_diff);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // clean up
  free(labels);
  free(predictions);
  cudaFree(losses);
  cudaFree(diffs);
  cudaFree(triplets_device);

  return d.ok();
}


template <typename Dtype>
__global__ void TripletBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}

 
bool TripletBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  TripletBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
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
