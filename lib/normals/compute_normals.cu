#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <Eigen/Eigen>

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) 
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

void set_device(int device_id)
{
  int current_device;
  checkCuda(cudaGetDevice(&current_device));
  if (current_device == device_id)
    return;
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  checkCuda(cudaSetDevice(device_id));
}

__global__ void computeVmapKernel(float* depth, float* vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff, int height, int width)
{
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  int u = threadIdx.y + blockIdx.y * blockDim.y;

  if(u < height && v < width)
  {
    float z = depth[u * width + v];

    if(z != 0 && z < depthCutoff)
    {
      float vx = z * (u - cx) * fx_inv;
      float vy = z * (v - cy) * fy_inv;
      float vz = z;

      vmap[0 + 3 * (u * width + v)] = vx;
      vmap[1 + 3 * (u * width + v)] = vy;
      vmap[2 + 3 * (u * width + v)] = vz;
    }
    else
    {
      vmap[0 + 3 * (u * width + v)] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
      vmap[1 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
      vmap[2 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
    }
  }
}

__global__ void computeNmapKernel(float* vmap, float* nmap, int height, int width)
{
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  int u = threadIdx.y + blockIdx.y * blockDim.y;

  if(u >= height || v >= width)
    return;

  if(u == height - 1 || v == width - 1)
  {
    nmap[0 + 3 * (u * width + v)] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
    nmap[1 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
    nmap[2 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
    return;
  }

  Eigen::Matrix<float,3,1,Eigen::DontAlign> v00, v01, v10;
  v00(0) = vmap[0 + 3 * (u * width + v)];
  v01(0) = vmap[0 + 3 * ((u + 1) * width + v)];
  v10(0) = vmap[0 + 3 * (u * width + v + 1)];

  if(!isnan(v00(0)) && !isnan(v01(0)) && !isnan(v10(0)))
  {
    v00(1) = vmap[1 + 3 * (u * width + v)];
    v01(1) = vmap[1 + 3 * ((u + 1) * width + v)];
    v10(1) = vmap[1 + 3 * (u * width + v + 1)];

    v00(2) = vmap[2 + 3 * (u * width + v)];
    v01(2) = vmap[2 + 3 * ((u + 1) * width + v)];
    v10(2) = vmap[2 + 3 * (u * width + v + 1)];

    Eigen::Matrix<float,3,1,Eigen::DontAlign> r = (v01 - v00).cross(v10 - v00).normalized();

    nmap[0 + 3 * (u * width + v)] = r(0);
    nmap[1 + 3 * (u * width + v)] = r(1);
    nmap[2 + 3 * (u * width + v)] = r(2);
  }
  else
  {
    nmap[0 + 3 * (u * width + v)] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
    nmap[1 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
    nmap[2 + 3 * (u * width + v)] = __int_as_float(0x7fffffff);
  }
}

void compute_normals(float* depth, float* nmap, float fx, float fy, float cx, float cy, float depthCutoff, int height, int width, int device_id)
{
  set_device(device_id);

  // allocate memory
  float* depth_device;
  float* vmap_device;
  float* nmap_device;

  checkCuda(cudaMalloc((void **)&depth_device, height * width * sizeof(float)));
  checkCuda(cudaMalloc((void **)&vmap_device, height * width * 3 * sizeof(float)));
  checkCuda(cudaMalloc((void **)&nmap_device, height * width * 3 * sizeof(float)));
  checkCuda(cudaMemcpy(depth_device, depth, height * width * sizeof(float), cudaMemcpyHostToDevice));

  // compute vmap
  dim3 block(32, 8);
  dim3 grid(1, 1, 1);
  grid.x = divUp(width, block.x);
  grid.y = divUp(height, block.y);

  computeVmapKernel<<<grid, block>>>(depth_device, vmap_device, 1.f / fx, 1.f / fy, cx, cy, depthCutoff, height, width);
  checkCuda(cudaGetLastError());

  // compute nmap
  computeNmapKernel<<<grid, block>>>(vmap_device, nmap_device, height, width);
  checkCuda(cudaGetLastError());

  // copy output
  checkCuda(cudaMemcpy(nmap, nmap_device, height * width * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // clean up
  checkCuda(cudaFree(depth_device));
  checkCuda(cudaFree(vmap_device));
  checkCuda(cudaFree(nmap_device));
}
