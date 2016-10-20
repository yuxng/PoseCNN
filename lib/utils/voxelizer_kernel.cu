// --------------------------------------------------------
// FCN
// Copyright (c) 2016
// Licensed under The MIT License [see LICENSE for details]
// Written by Yu Xiang
// --------------------------------------------------------

#include <vector>
#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void voxel_kernel(const int nthreads, const int grid_size, const float step_d, const float step_h, const float step_w,
                   const float min_d, const float min_h, const float min_w,
                   const int filter_h, const int filter_w, const int num_classes,
                   const int height, const int width, const int* grid_indexes, const int* labels,
                   const float* pmatrix, int* top_locations, int* top_label_count, int* top_labels)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (d, h, w) is an element in the output
    int n = index;
    int w = n % grid_size;
    n /= grid_size;
    int h = n % grid_size;
    int d = n / grid_size;

    // voxel location in 3D
    float X = d * step_d + min_d;
    float Y = h * step_h + min_h;
    float Z = w * step_w + min_w;

    // project the 3D point to image
    float x1 = pmatrix[0] * X + pmatrix[3] * Y + pmatrix[6] * Z + pmatrix[9];
    float x2 = pmatrix[1] * X + pmatrix[4] * Y + pmatrix[7] * Z + pmatrix[10];
    float x3 = pmatrix[2] * X + pmatrix[5] * Y + pmatrix[8] * Z + pmatrix[11];
    int px = int(x1 / x3);
    int py = int(x2 / x3);

    // reset the label counter
    for (int i = 0; i < num_classes; i++)
      top_label_count[index * num_classes + i] = 0;

    // reset the pixel locations
    for (int i = 0; i < filter_w * filter_h; i++)
      top_locations[index * filter_w * filter_h + i] = -1;

    // loop over a neighborhood around the pixel
    int count = 0;
    for (int i = 0; i < filter_w; i++)
    {
      int x = px + i - filter_w / 2;
      if (x >= 0 && x < width)
      {
        for (int j = 0; j < filter_h; j++)
        {
          int y = py + j - filter_h / 2;
          if (y >= 0 && y < height)
          {
            // pixel falling into the voxel
            if (grid_indexes[y * width + x] == index)
            {
              // store the pixel locatoin
              top_locations[index * filter_w * filter_h + count] = y * width + x;
              count++;
              // count the pixel label
              int label = labels[y * width + x];
              top_label_count[index * num_classes + label]++;
            }
          }
        }
      }
    }
    // find the class label
    int num = -1;
    int label = 0;
    for (int i = 0; i < num_classes; i++)
    {
      if (top_label_count[index * num_classes + i] > num)
      {
        num = top_label_count[index * num_classes + i];
        label = i;
      }
    }
    top_labels[index] = label;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _build_voxels(const int grid_size, const float step_d, const float step_h, const float step_w,
                   const float min_d, const float min_h, const float min_w,
                   const int filter_h, const int filter_w, const int num_classes,
                   const int height, const int width, const int* grid_indexes, const int* labels,
                   const float* pmatrix, int* top_locations, int* top_labels, int device_id)
{
  _set_device(device_id);
  const int kThreadsPerBlock = 1024;

  // output
  int* top_locations_dev = NULL;
  int* top_labels_dev = NULL;
  int* top_label_count = NULL;
  CUDA_CHECK(cudaMalloc(&top_locations_dev,
                        grid_size * grid_size * grid_size * filter_h * filter_w * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&top_labels_dev,
                        grid_size * grid_size * grid_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&top_label_count,
                        grid_size * grid_size * grid_size * num_classes * sizeof(int)));

  // internal usage
  int* grid_indexes_dev = NULL;
  int* labels_dev = NULL;
  CUDA_CHECK(cudaMalloc(&grid_indexes_dev, height * width * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&labels_dev, height * width * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(grid_indexes_dev,
                        grid_indexes,
                        height * width * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(labels_dev,
                        labels,
                        height * width * sizeof(int),
                        cudaMemcpyHostToDevice));

  float* pmatrix_dev = NULL;
  CUDA_CHECK(cudaMalloc(&pmatrix_dev, 12 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(pmatrix_dev,
                        pmatrix,
                        12 * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int output_size = grid_size * grid_size * grid_size;
  voxel_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock>>>(output_size, grid_size, step_d, step_h, step_w, min_d, min_h, min_w, filter_h, filter_w, num_classes, height, width, 
                                           grid_indexes_dev, labels_dev, pmatrix_dev, top_locations_dev, top_label_count, top_labels_dev);

  CUDA_CHECK(cudaMemcpy(top_locations,
                        top_locations_dev,
                        grid_size * grid_size * grid_size * filter_h * filter_w * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(top_labels,
                        top_labels_dev,
                        grid_size * grid_size * grid_size * sizeof(int),
                        cudaMemcpyDeviceToHost));


  CUDA_CHECK(cudaFree(top_locations_dev));
  CUDA_CHECK(cudaFree(top_labels_dev));
  CUDA_CHECK(cudaFree(top_label_count));
  CUDA_CHECK(cudaFree(grid_indexes_dev));
  CUDA_CHECK(cudaFree(labels_dev));
  CUDA_CHECK(cudaFree(pmatrix_dev));
}
