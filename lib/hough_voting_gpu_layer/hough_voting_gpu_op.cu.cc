#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include <time.h>
#include <thrust/extrema.h>
#include <Eigen/Geometry> 
#include <cublas_v2.h>
#include "hough_voting_gpu_op.h"

#define VERTEX_CHANNELS 3
#define MAX_ROI 128

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

__device__ inline float point2line(int cx, int cy, int x, int y, float u, float v)
{
  float n1 = -v;
  float n2 = u;

  return fabs(n1 * (cx - x) + n2 * (cy - y)) / sqrt(n1 * n1 + n2 * n2);
}


__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  return distance;
}

__device__ inline float angle_distance_label(int cx, int cy, int x, int y, float u, float v, 
  int cls, const int height, const int width, const int* labelmap)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  int num = 20;
  int count = 0;
  for (int i = 1; i <= num; i++)
  {
    float step = float(i) / float(num);
    int px = int(x + step * dx);
    int py = int(y + step * dy);
    if (px >= 0 && px < width && py >= 0 && py < height)
    {
      if (labelmap[py * width + px] == cls)
        count++;
    }
  }
  if ((float)count / float(num) < 0.8)
    distance = 0;

  return distance;
}

__device__ inline float IoU(float* a, float* b) 
{
  float left = fmax(a[0], b[0]), right = fmin(a[2], b[2]);
  float top = fmax(a[1], b[1]), bottom = fmin(a[3], b[3]);
  float width = fmax(right - left + 1, 0.f), height = fmax(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__device__ inline void project_box(int cls, const float* extents, const float* meta_data, float distance, float factor, float* threshold)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;
  float bb3D[24];

  bb3D[0] = xHalf; bb3D[1] = yHalf; bb3D[2] = zHalf + distance;
  bb3D[3] = -xHalf; bb3D[4] = yHalf; bb3D[5] = zHalf + distance;
  bb3D[6] = xHalf; bb3D[7] = -yHalf; bb3D[8] = zHalf + distance;
  bb3D[9] = -xHalf; bb3D[10] = -yHalf; bb3D[11] = zHalf + distance;
  bb3D[12] = xHalf; bb3D[13] = yHalf; bb3D[14] = -zHalf + distance;
  bb3D[15] = -xHalf; bb3D[16] = yHalf; bb3D[17] = -zHalf + distance;
  bb3D[18] = xHalf; bb3D[19] = -yHalf; bb3D[20] = -zHalf + distance;
  bb3D[21] = -xHalf; bb3D[22] = -yHalf; bb3D[23] = -zHalf + distance;

  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float minX = 1e8;
  float maxX = -1e8;
  float minY = 1e8;
  float maxY = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float x = fx * (bb3D[i * 3] / bb3D[i * 3 + 2])  + px;
    float y = fy * (bb3D[i * 3 + 1] / bb3D[i * 3 + 2])  + py;
    minX = fmin(minX, x);
    minY = fmin(minY, y);
    maxX = fmax(maxX, x);
    maxY = fmax(maxY, y);
  }
  float width = maxX - minX + 1;
  float height = maxY - minY + 1;
  *threshold = fmax(width, height) * factor;
}


__device__ inline float compute_box_overlap(int cls, const float* extents, const float* meta_data, const float* pose, float* box)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;

  Eigen::Matrix<float,8,3,Eigen::DontAlign> bb3D;
  bb3D(0, 0) = xHalf; bb3D(0, 1) = yHalf; bb3D(0, 2) = zHalf;
  bb3D(1, 0) = -xHalf; bb3D(1, 1) = yHalf; bb3D(1, 2) = zHalf;
  bb3D(2, 0) = xHalf; bb3D(2, 1) = -yHalf; bb3D(2, 2) = zHalf;
  bb3D(3, 0) = -xHalf; bb3D(3, 1) = -yHalf; bb3D(3, 2) = zHalf;
  bb3D(4, 0) = xHalf; bb3D(4, 1) = yHalf; bb3D(4, 2) = -zHalf;
  bb3D(5, 0) = -xHalf; bb3D(5, 1) = yHalf; bb3D(5, 2) = -zHalf;
  bb3D(6, 0) = xHalf; bb3D(6, 1)= -yHalf; bb3D(6, 2) = -zHalf;
  bb3D(7, 0) = -xHalf; bb3D(7, 1) = -yHalf; bb3D(7, 2) = -zHalf;

  // rotation
  Eigen::Quaternionf quaternion(pose[6], pose[7], pose[8], pose[9]);
  Eigen::Matrix3f rmatrix = quaternion.toRotationMatrix();
  Eigen::Matrix<float,3,8,Eigen::DontAlign> bb3D_new = rmatrix * bb3D.transpose();

  // projection
  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float x1 = 1e8;
  float x2 = -1e8;
  float y1 = 1e8;
  float y2 = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float X = bb3D_new(0, i) + pose[10];
    float Y = bb3D_new(1, i) + pose[11];
    float Z = bb3D_new(2, i) + pose[12];
    float x = fx * (X / Z)  + px;
    float y = fy * (Y / Z)  + py;
    x1 = fmin(x1, x);
    y1 = fmin(y1, y);
    x2 = fmax(x2, x);
    y2 = fmax(y2, y);
  }

  float box_gt[4];
  box_gt[0] = x1;
  box_gt[1] = y1;
  box_gt[2] = x2;
  box_gt[3] = y2;
  return IoU(box, box_gt);
}

__global__ void compute_arrays_kernel(const int nthreads, const int* labelmap,
    int* arrays, int* array_size, const int height, const int width) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int cls = labelmap[index];
    if (cls > 0)
    {
      int size = atomicAdd(array_size + cls, 1);
      int offset = cls * height * width + size;
      arrays[offset] = index;
    }
  }
}

/*
__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, const int* labelmap, 
    const float* vertmap, const float* extents, const float* meta_data, int* arrays, int* array_size, 
    int* class_indexes, const int height, const int width, const int num_classes, const int count, const float inlierThreshold, const int skip_pixels) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (cls, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int cls = class_indexes[ind];
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;

    int size = array_size[cls];
    float distance = 0;
    float bb_width = -1;
    float bb_height = -1;
    float threshold;
    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = cls * height * width + i;
      int location = arrays[offset];
      int x = location % width;
      int y = location / width;

      // read the direction
      offset = VERTEX_CHANNELS * cls + VERTEX_CHANNELS * num_classes * (y * width + x);
      float u = vertmap[offset];
      float v = vertmap[offset + 1];
      float d = exp(vertmap[offset + 2]);

      // vote
      if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
      // if (point2line(cx, cy, x, y, u, v) < 1 && angle_distance_label(cx, cy, x, y, u, v, cls, height, width, labelmap) > 0)
      {
        project_box(cls, extents, meta_data, d, 0.6, &threshold);
        float dx = fabsf(x - cx);
        float dy = fabsf(y - cy);
        if (dx < threshold && dy < threshold)
        {
          hough_space[index]++;
          distance += d;
        }
        if (dx > bb_width && dx < threshold && dy < threshold)
          bb_width = dx;
        if (dy > bb_height && dx < threshold && dy < threshold)
          bb_height = dy;
      }
    }

    if (hough_space[index] > 0)
    {
      distance /= hough_space[index];
      int offset = ind * height * width * 3 + 3 * (cy * width + cx);
      hough_data[offset] = distance;
      hough_data[offset + 1] = 2 * bb_height;
      hough_data[offset + 2] = 2 * bb_width;
    }
  }
}

*/

__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, const int* labelmap, 
    const float* vertmap, const float* extents, const float* meta_data, int* arrays, int* array_size, 
    int* class_indexes, const int height, const int width, const int num_classes, const int count, const float inlierThreshold, const int skip_pixels) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (cls, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int cls = class_indexes[ind];
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int size = array_size[cls];
    float distance = 0;
    float threshold;

    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = cls * height * width + i;
      int location = arrays[offset];
      int x = location % width;
      int y = location / width;

      // read the direction
      offset = VERTEX_CHANNELS * cls + VERTEX_CHANNELS * num_classes * (y * width + x);
      float u = vertmap[offset];
      float v = vertmap[offset + 1];
      float d = exp(vertmap[offset + 2]);

      // vote
      if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
      {
        project_box(cls, extents, meta_data, d, 0.6, &threshold);
        float dx = fabsf(x - cx);
        float dy = fabsf(y - cy);
        if (dx < threshold && dy < threshold)
        {
          hough_space[index]++;
          distance += d;
        }
      }
    }

    if (hough_space[index] > 0)
    {
      distance /= hough_space[index];

      float bb_width = -1;
      float bb_height = -1;
      for (int i = 0; i < size; i += skip_pixels)
      {
        int offset = cls * height * width + i;
        int location = arrays[offset];
        int x = location % width;
        int y = location / width;

        // read the direction
        offset = VERTEX_CHANNELS * cls + VERTEX_CHANNELS * num_classes * (y * width + x);
        float u = vertmap[offset];
        float v = vertmap[offset + 1];

        // vote
        if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
        {
          project_box(cls, extents, meta_data, distance, 0.6, &threshold);
          float dx = fabsf(x - cx);
          float dy = fabsf(y - cy);
          if (dx > bb_width && dx < threshold && dy < threshold)
            bb_width = dx;
          if (dy > bb_height && dx < threshold && dy < threshold)
            bb_height = dy;
        }
      }

      int offset = ind * height * width * 3 + 3 * (cy * width + cx);
      hough_data[offset] = distance;
      hough_data[offset + 1] = 2 * bb_height;
      hough_data[offset + 2] = 2 * bb_width;
    }
  }
}

__global__ void compute_max_indexes_kernel(const int nthreads, int* max_indexes, int index_size, int* num_max, float* hough_space, 
  float* hough_data, int height, int width, float threshold, float perThreshold)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (ind, cx, cy) is an element in the hough space
    int ind = index / (height * width);
    int n = index % (height * width);
    int cx = n % width;
    int cy = n / width;
    int kernel_size = 3;

    int offset = ind * height * width * 3 + 3 * (cy * width + cx);
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (hough_space[index] > threshold && bb_height > 0 && bb_width > 0)
    {
      // check if the location is local maximum
      int flag = 0;
      for (int x = cx - kernel_size; x <= cx + kernel_size; x++)
      {
        for (int y = cy - kernel_size; y <= cy + kernel_size; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            if (hough_space[ind * height * width + y * width + x] > hough_space[index])
            {
              flag = 1;
              break;
            }
          }
        }

        // check the percentage of voting
        if (hough_space[index] / (bb_height * bb_width) < perThreshold)
          flag = 1;
      }

      if (flag == 0)
      {
        // add the location to max_indexes
        int max_index = atomicAdd(num_max, 1);
        if (max_index < index_size)
          max_indexes[max_index] = index;
      }
    }
  }
}


__global__ void compute_rois_kernel(const int nthreads, float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain,
    const float* extents, const float* meta_data, const float* gt, float* hough_space, float* hough_data, int* max_indexes, int* class_indexes,
    int is_train, int batch_index, const int height, const int width, const int num_classes, const int num_gt, int* num_rois) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    float scale = 0.05;
    int max_index = max_indexes[index];
    int ind = max_index / (height * width);
    int cls = class_indexes[ind];
    int n = max_index % (height * width);
    int x = n % width;
    int y = n / width;

    float fx = meta_data[0];
    float fy = meta_data[4];
    float px = meta_data[2];
    float py = meta_data[5];
    float rx = (x - px) / fx;
    float ry = (y - py) / fy;

    int offset = ind * height * width * 3 + 3 * (y * width + x);
    float bb_distance = hough_data[offset];
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    if (is_train)
    {
      int roi_index = atomicAdd(num_rois, 9);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      for (int i = 0; i < 9; i++)
      {
        top_pose[(roi_index + i) * 7 + 0] = 1;
        top_pose[(roi_index + i) * 7 + 1] = 0;
        top_pose[(roi_index + i) * 7 + 2] = 0;
        top_pose[(roi_index + i) * 7 + 3] = 0;
        top_pose[(roi_index + i) * 7 + 4] = rx * bb_distance;
        top_pose[(roi_index + i) * 7 + 5] = ry * bb_distance;
        top_pose[(roi_index + i) * 7 + 6] = bb_distance;

        if (num_gt == 0)
          top_domain[roi_index + i] = 1;
        else
          top_domain[roi_index + i] = 0;
      }

      // compute pose target
      for (int i = 0; i < num_gt; i++)
      {
        int gt_batch = int(gt[i * 13 + 0]);
        int gt_id = int(gt[i * 13 + 1]);
        if(cls == gt_id && batch_index == gt_batch)
        {
          int gt_ind = i;

          float overlap = compute_box_overlap(cls, extents, meta_data, gt + gt_ind * 13, top_box + roi_index * 7 + 2);
          if (overlap > 0.2)
          {
            for (int j = 0; j < 9; j++)
            {
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 0] = gt[gt_ind * 13 + 6];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 1] = gt[gt_ind * 13 + 7];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 2] = gt[gt_ind * 13 + 8];
              top_target[(roi_index + j) * 4 * num_classes + 4 * cls + 3] = gt[gt_ind * 13 + 9];

              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 0] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 1] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 2] = 1;
              top_weight[(roi_index + j) * 4 * num_classes + 4 * cls + 3] = 1;
            }
            break;
          }
        }
      }

      // add jittering boxes
      float x1 = top_box[roi_index * 7 + 2];
      float y1 = top_box[roi_index * 7 + 3];
      float x2 = top_box[roi_index * 7 + 4];
      float y2 = top_box[roi_index * 7 + 5];
      float ww = x2 - x1;
      float hh = y2 - y1;

      // (-1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (-1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (0, -1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 - 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (-1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 - 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (0, +1)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1;
      top_box[roi_index * 7 + 3] = y1 + 0.05 * hh;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      // (+1, 0)
      roi_index++;
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x1 + 0.05 * ww;
      top_box[roi_index * 7 + 3] = y1;
      top_box[roi_index * 7 + 4] = top_box[roi_index * 7 + 2] + ww;
      top_box[roi_index * 7 + 5] = top_box[roi_index * 7 + 3] + hh;
      top_box[roi_index * 7 + 6] = hough_space[max_index];
    }
    else
    {
      int roi_index = atomicAdd(num_rois, 1);
      top_box[roi_index * 7 + 0] = batch_index;
      top_box[roi_index * 7 + 1] = cls;
      top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
      top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
      top_box[roi_index * 7 + 6] = hough_space[max_index];

      top_pose[roi_index * 7 + 0] = 1;
      top_pose[roi_index * 7 + 1] = 0;
      top_pose[roi_index * 7 + 2] = 0;
      top_pose[roi_index * 7 + 3] = 0;
      top_pose[roi_index * 7 + 4] = rx * bb_distance;
      top_pose[roi_index * 7 + 5] = ry * bb_distance;
      top_pose[roi_index * 7 + 6] = bb_distance;
    }
  }
}


void reset_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, int num_classes)
{
  int num = MAX_ROI * 9;
  cudaMemset(top_box, 0, num * 7 * sizeof(float));
  cudaMemset(top_pose, 0, num * 7 * sizeof(float));
  cudaMemset(top_target, 0, num * 4 *num_classes * sizeof(float));
  cudaMemset(top_weight, 0, num * 4 * num_classes * sizeof(float));
  cudaMemset(top_domain, 0, num * sizeof(int));
  cudaMemset(num_rois, 0, sizeof(int));
}


void copy_num_rois(int* num_rois, int* num_rois_device)
{
  cudaMemcpy(num_rois, num_rois_device, sizeof(int), cudaMemcpyDeviceToHost);
}


void copy_outputs(float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain,
  float* top_box_final, float* top_pose_final, float* top_target_final, float* top_weight_final, int* top_domain_final, int num_classes, int num_rois)
{
  cudaMemcpy(top_box_final, top_box, num_rois * 7 * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(top_pose_final, top_pose, num_rois * 7 * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(top_target_final, top_target, num_rois * 4 * num_classes * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(top_weight_final, top_weight, num_rois * 4 * num_classes * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(top_domain_final, top_domain, num_rois * sizeof(int), cudaMemcpyDeviceToDevice);
}


void set_gradients(float* top_label, float* top_vertex, int batch_size, int height, int width, int num_classes)
{
  cudaMemset(top_label, 0, batch_size * height * width * sizeof(float));
  cudaMemset(top_vertex, 0, batch_size * height * width * 3 * num_classes * sizeof(float));
}


void HoughVotingLaucher(OpKernelContext* context,
    const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* gt,
    const int batch_index, const int batch_size, const int height, const int width, const int num_classes, const int num_gt, 
    const int is_train, const float inlierThreshold, const int labelThreshold, const float votingThreshold, const float perThreshold, 
    const int skip_pixels, 
    float* top_box, float* top_pose, float* top_target, float* top_weight, int* top_domain, int* num_rois, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  int output_size;
  cudaError_t err;

  // step 1: compute a label index array for each class
  int dims[2];
  dims[0] = num_classes;
  dims[1] = height * width;
  TensorShape output_shape_arrays;
  TensorShapeUtils::MakeShape(dims, 2, &output_shape_arrays);
  Tensor arrays_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_arrays, &arrays_tensor));
  int* arrays = arrays_tensor.flat<int>().data();

  TensorShape output_shape_array_sizes;
  TensorShapeUtils::MakeShape(&num_classes, 1, &output_shape_array_sizes);
  Tensor array_sizes_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_array_sizes, &array_sizes_tensor));
  int* array_sizes = array_sizes_tensor.flat<int>().data();
  cudaMemset(array_sizes, 0, num_classes * sizeof(int));

  output_size = height * width;
  compute_arrays_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, labelmap, arrays, array_sizes, height, width);
  cudaThreadSynchronize();

  // compute class indexes
  int* array_sizes_host = (int*)malloc(num_classes * sizeof(int));
  int* class_indexes_host = (int*)malloc(num_classes * sizeof(int));
  cudaMemcpy(array_sizes_host, array_sizes, num_classes * sizeof(int), cudaMemcpyDeviceToHost);
  int count = 0;
  for (int c = 1; c < num_classes; c++)
  {
    if (array_sizes_host[c] > labelThreshold)
    {
      class_indexes_host[count] = c;
      count++;
    }
    // else
    //  printf("class %d with only pixels %d\n", c, array_sizes_host[c]);
  }

  if (count == 0)
  {
    free(array_sizes_host);
    free(class_indexes_host);
    return;
  }

  TensorShape output_shape_class_indexes;
  TensorShapeUtils::MakeShape(&count, 1, &output_shape_class_indexes);
  Tensor class_indexes_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_class_indexes, &class_indexes_tensor));
  int* class_indexes = class_indexes_tensor.flat<int>().data();
  cudaMemcpy(class_indexes, class_indexes_host, count * sizeof(int), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute label index: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 2: compute the hough space
  int hdims[4];
  hdims[0] = count;
  hdims[1] = height;
  hdims[2] = width;
  hdims[3] = 1;
  TensorShape output_shape_hough_space;
  TensorShapeUtils::MakeShape(hdims, 4, &output_shape_hough_space);
  Tensor hough_space_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_hough_space, &hough_space_tensor));
  float* hough_space = hough_space_tensor.flat<float>().data(); 
  if (cudaMemset(hough_space, 0, count * height * width * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  hdims[3] = 3;
  TensorShape output_shape_hough_data;
  TensorShapeUtils::MakeShape(hdims, 4, &output_shape_hough_data);
  Tensor hough_data_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output_shape_hough_data, &hough_data_tensor));
  float* hough_data = hough_data_tensor.flat<float>().data(); 
  if (cudaMemset(hough_data, 0, count * height * width * 3 * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  output_size = count * height * width;
  compute_hough_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, hough_space, hough_data, labelmap, vertmap, extents, meta_data,
      arrays, array_sizes, class_indexes, height, width, num_classes, count, inlierThreshold, skip_pixels);
  cudaThreadSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute hough space: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 3: find the maximum in hough space
  int dim = 1;
  TensorShape output_shape_num_max;
  TensorShapeUtils::MakeShape(&dim, 1, &output_shape_num_max);
  Tensor num_max_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_num_max, &num_max_tensor));
  int* num_max = num_max_tensor.flat<int>().data();
  if (cudaMemset(num_max, 0, sizeof(int)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  int index_size = MAX_ROI / batch_size;
  TensorShape output_shape_max_indexes;
  TensorShapeUtils::MakeShape(&index_size, 1, &output_shape_max_indexes);
  Tensor max_indexes_tensor;
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, output_shape_max_indexes, &max_indexes_tensor));
  int* max_indexes = max_indexes_tensor.flat<int>().data(); 
  if (cudaMemset(max_indexes, 0, index_size * sizeof(int)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  if (votingThreshold > 0)
  {
    output_size = count * height * width;
    compute_max_indexes_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, max_indexes, index_size, num_max, hough_space, hough_data, height, width, votingThreshold, perThreshold);
    cudaThreadSynchronize();
  }
  else
  {
    int* max_indexes_host = (int*)malloc(count * sizeof(int));
    memset(max_indexes_host, 0, count * sizeof(int));
    for (int i = 0; i < count; i++)
    {
      float *hmax = thrust::max_element(thrust::device, hough_space + i * height * width, hough_space + (i+1) * height * width);
      max_indexes_host[i] = hmax - hough_space;
    }
    cudaMemcpy(num_max, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_indexes, max_indexes_host, count * sizeof(int), cudaMemcpyHostToDevice);
    free(max_indexes_host);
  }
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute maximum: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 4: compute outputs
  int num_max_host;
  cudaMemcpy(&num_max_host, num_max, sizeof(int), cudaMemcpyDeviceToHost);
  if (num_max_host >= index_size)
    num_max_host = index_size;
  // printf("num_max: %d\n", num_max_host);
  if (num_max_host > 0)
  {
    output_size = num_max_host;
    compute_rois_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                         kThreadsPerBlock, 0, d.stream()>>>(
        output_size, top_box, top_pose, top_target, top_weight, top_domain,
        extents, meta_data, gt, hough_space, hough_data, max_indexes, class_indexes,
        is_train, batch_index, height, width, num_classes, num_gt, num_rois);
    cudaThreadSynchronize();
  }
  
  // clean up
  free(array_sizes_host);
  free(class_indexes_host);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute outputs: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
