#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include "lifted_structured_loss_op_gpu.h"

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
__global__ void compute_distance_label_matrix(const int nthreads, const Dtype* bottom_data, const int* pixel_indexes, const int* pixel_labels,
    const int channels, const int num_pixels, Dtype* dis_mat, bool* label_mat) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (i, j) is the pair to consider
    int i = index / num_pixels;
    int j = index % num_pixels;

    int index_i = pixel_indexes[i];
    int index_j = pixel_indexes[j];

    const Dtype* pi = bottom_data + index_i * channels;
    const Dtype* pj = bottom_data + index_j * channels;

    // distance
    Dtype dis = 0;
    for (int c = 0; c < channels; c++)
      dis += (pi[c] - pj[c]) * (pi[c] - pj[c]);
    dis_mat[index] = sqrt(dis);
    
    // label
    label_mat[index] = (pixel_labels[i] == pixel_labels[j]);
  }
}

template <typename Dtype>
__global__ void LiftedstructForward(const int nthreads, const Dtype* bottom_data, const int* pixel_indexes, const int* positive_indexes, const Dtype* dis_mat, const bool* label_mat,
    const int channels, const int num_pixels, const float margin, Dtype* losses, Dtype* diffs) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (i, j) is the pair to consider
    int num_positives = nthreads;
    int pos_index = positive_indexes[index];
    int i = pos_index / num_pixels;
    int j = pos_index % num_pixels;

    int index_i = pixel_indexes[i];
    int index_j = pixel_indexes[j];

    Dtype dist_pos = dis_mat[i * num_pixels + j];

    if(label_mat[i * num_pixels + j] && dist_pos > 0)
    {
      // 1.count the number of negatives for this positive
      int num_negatives = 0;
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[i * num_pixels + k] && dis_mat[i * num_pixels + k] > 0)
          num_negatives++;
      }
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[j * num_pixels + k] && dis_mat[j * num_pixels + k] > 0)
          num_negatives++;
      }

      // 2. compute loss augmented inference
      Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> loss_aug_inference(num_negatives, 1);
      int neg_idx = 0;

      // mine negative (anchor i, neg k)
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[i * num_pixels + k] && dis_mat[i * num_pixels + k] > 0)
        {
          loss_aug_inference(neg_idx, 0) = margin - dis_mat[i * num_pixels + k];
          neg_idx++;
        }
      }

      // mine negative (anchor j, neg k)
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[j * num_pixels + k] && dis_mat[j * num_pixels + k] > 0)
        {
          loss_aug_inference(neg_idx, 0) = margin - dis_mat[j * num_pixels + k];
          neg_idx++;
        }
      }

      // compute softmax of loss aug inference vector;
      Dtype max_elem = loss_aug_inference.maxCoeff();
      loss_aug_inference = (loss_aug_inference.array() - max_elem).exp();
      Dtype sum_exp = loss_aug_inference.sum();
      Dtype soft_maximum = log(sum_exp) + max_elem;

      // hinge the soft_maximum - S_ij (positive pair similarity)
      Dtype this_loss = max(soft_maximum + dist_pos, Dtype(0.0));

      // squared hinge
      losses[index] = this_loss * this_loss; 

      // 3. compute gradients

      // update from positive distance dJ_dD_{ij}; update x_i, x_j
      Dtype scaler = 2.0 * this_loss / dist_pos;

      // update x_i
      for (int c = 0; c < channels; c++)
      {
        int ind = i * channels * num_positives + c * num_positives + index;
        diffs[ind] += scaler * (bottom_data[index_i * channels + c] - bottom_data[index_j * channels + c]);
      }

      // update x_j
      for (int c = 0; c < channels; c++)
      {
        int ind = j * channels * num_positives + c * num_positives + index;
        diffs[ind] += -scaler * (bottom_data[index_i * channels + c] - bottom_data[index_j * channels + c]);
      }

      // update from negative distance dJ_dD_{ik}; update x_i, x_k
      neg_idx = 0;
      Dtype dJ_dDik = 0;
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[i * num_pixels + k] && dis_mat[i * num_pixels + k] > 0)
        {
          int index_k = pixel_indexes[k];

          dJ_dDik = 2.0 * this_loss * (-1.0) * loss_aug_inference(neg_idx, 0) / sum_exp;
          neg_idx++;

          scaler = dJ_dDik / dis_mat[i * num_pixels + k];

          // update x_i
          for (int c = 0; c < channels; c++)
          {
            int ind = i * channels * num_positives + c * num_positives + index;
            diffs[ind] += scaler * (bottom_data[index_i * channels + c] - bottom_data[index_k * channels + c]);
          }

          // update x_k
          for (int c = 0; c < channels; c++)
          {
            int ind = k * channels * num_positives + c * num_positives + index;
            diffs[ind] += -scaler * (bottom_data[index_i * channels + c] - bottom_data[index_k * channels + c]);
          }
        }
      }

      // update from negative distance dJ_dD_{jk}; update x_j, x_k
      Dtype dJ_dDjk = 0;
      for (int k = 0; k < num_pixels; k++)
      {
        if (!label_mat[j * num_pixels + k] && dis_mat[j * num_pixels + k] > 0)
        {
          int index_k = pixel_indexes[k];

          dJ_dDjk = 2.0 * this_loss * (-1.0) * loss_aug_inference(neg_idx, 0) / sum_exp;
          neg_idx++;

          scaler = dJ_dDjk / dis_mat[j * num_pixels + k];

          // update x_j
          for (int c = 0; c < channels; c++)
          {
            int ind = j * channels * num_positives + c * num_positives + index;
            diffs[ind] += scaler * (bottom_data[index_j * channels + c] - bottom_data[index_k * channels + c]);
          }

          // update x_k
          for (int c = 0; c < channels; c++)
          {
            int ind = k * channels * num_positives + c * num_positives + index;
            diffs[ind] += -scaler * (bottom_data[index_j * channels + c] - bottom_data[index_k * channels + c]);
          }
        }
      }
    }
    else
    {
      losses[index] = 0; 
    }
  }
}

template <typename Dtype>
__global__ void sum_gradients(const int nthreads, const Dtype* diffs, const int* pixel_indexes,
    const int channels, const int num_positives, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (i, j) is the pair to consider
    int n = index / channels;
    int c = index % channels;

    Dtype diff_sum = 0;
    for(int i = 0; i < num_positives; i++)
    {
      int ind = n * channels * num_positives + c * num_positives + i;
      diff_sum += diffs[ind];
    }

    bottom_diff[pixel_indexes[n] * channels + c] = diff_sum / num_positives / 2.0;
  }
}

// bottom_data: (batch_size, height, width, channels)
bool LiftedstructForwardLaucher(
    const float* bottom_data, const float* bottom_label,
    const int batch_size, const int height, const int width, const int channels, const int num_classes,
    const float margin, const int budget, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d)
{
  // copy labels to CPU
  float* labels_host = (float*)malloc(batch_size * height * width * num_classes * sizeof(float));
  cudaMemcpy(labels_host, bottom_label, batch_size * height * width * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

  // sample pixels to define the loss
  // compute label indexes
  std::vector< std::vector<int> > label_indexes(num_classes);
  for (int n = 0; n < batch_size; n++)
  {
    for (int h = 0; h < height; h++)
    {
      for (int w = 0; w < width; w++)
      {
        int index = n * height * width + h * width + w;
        for (int c = 0; c < num_classes; c++)
        {
          if(labels_host[index * num_classes + c] > 0)
          {
            label_indexes[c].push_back(index);
            break;
          }
        }
      } 
    }
  }

  // number of classes in the batch
  int count_classes = 0;
  for (int i = 0; i < num_classes; i++)
  {
    if (label_indexes[i].size() > 0)
    {
      count_classes++;
    }
  }

  // decide how many pixels to sample for each class
  int num_pixels_per_class = budget / count_classes;

  // sampling
  std::srand ( unsigned ( std::time(0) ) );
  std::vector<int> pixel_indexes;
  std::vector<int> pixel_labels;
  for (int i = 0; i < num_classes; i++)
  {
    // shuffle the indexes
    std::random_shuffle ( label_indexes[i].begin(), label_indexes[i].end() );
    for (int j = 0; j < label_indexes[i].size() && j < num_pixels_per_class; j++)
    {
      pixel_indexes.push_back(label_indexes[i][j]);
      pixel_labels.push_back(i);
    }
  }
  int num_pixels = pixel_indexes.size();

  // compute the indexes of positive pairs
  std::vector<int> positive_indexes;
  for (int i = 0; i < num_pixels; i++)
  {
    for (int j = i+1; j < num_pixels; j++)
    {
      if(pixel_labels[i] == pixel_labels[j])
        positive_indexes.push_back(i * num_pixels + j);
    }
  }
  int num_positives = positive_indexes.size();

  int* pixel_indexes_device;
  int* pixel_labels_device;
  int* positive_indexes_device;
  checkCuda(cudaMalloc(&pixel_indexes_device, num_pixels * sizeof(int))); 
  checkCuda(cudaMalloc(&pixel_labels_device, num_pixels * sizeof(int)));
  checkCuda(cudaMalloc(&positive_indexes_device, num_positives * sizeof(int)));
  cudaMemcpy(pixel_indexes_device, pixel_indexes.data(), num_pixels * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pixel_labels_device, pixel_labels.data(), num_pixels * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(positive_indexes_device, positive_indexes.data(), num_positives * sizeof(int), cudaMemcpyHostToDevice);

  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  // compute distance and label matrix
  const int output_size = num_pixels * num_pixels;
  float* dis_mat;
  bool* label_mat;
  checkCuda(cudaMalloc(&dis_mat, num_pixels * num_pixels * sizeof(float))); 
  checkCuda(cudaMalloc(&label_mat, num_pixels * num_pixels * sizeof(bool)));
  compute_distance_label_matrix<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, pixel_indexes_device, pixel_labels_device, channels, num_pixels, dis_mat, label_mat);
  cudaDeviceSynchronize();
  
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // compute loss matrix
  float* losses;
  float* diffs;
  checkCuda(cudaMalloc((void **) &losses, num_positives * sizeof(float))); 
  checkCuda(cudaMalloc((void **) &diffs, num_pixels * channels * num_positives *  sizeof(float))); 
  cudaMemset(diffs, 0, num_pixels * channels * num_positives *  sizeof(float));
  LiftedstructForward<<<(num_positives + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      num_positives, bottom_data, pixel_indexes_device, positive_indexes_device, dis_mat, label_mat, channels, num_pixels, margin, losses, diffs);
  cudaDeviceSynchronize();
  
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the loss and diffs
  thrust::device_ptr<float> losses_ptr(losses);
  float loss = thrust::reduce(losses_ptr, losses_ptr + num_positives);

  loss = loss / num_positives / 2.0;
  cudaMemcpy(top_data, &loss, sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(bottom_diff, 0, batch_size * height * width * channels *  sizeof(float));
  const int output_size_diff = num_pixels * channels;
  sum_gradients<<<(output_size_diff + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size_diff, diffs, pixel_indexes_device, channels, num_positives, bottom_diff);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  free(labels_host);
  cudaFree(dis_mat);
  cudaFree(label_mat);
  cudaFree(losses);
  cudaFree(diffs);
  cudaFree(pixel_indexes_device);
  cudaFree(pixel_labels_device);

  return d.ok();
}


template <typename Dtype>
__global__ void LiftedstructBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}

 
bool LiftedstructBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  LiftedstructBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
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
