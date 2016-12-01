#include <df/icpcuda/internal.h>
#include <df/icpcuda/safe_call.hpp>

#if __CUDA_ARCH__ < 300
__inline__ __device__
float __shfl_down(float val, int offset, int width = 32)
{
    static __shared__ float shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}
#endif

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

template<int D>
__inline__  __device__ void warpReduceSum(Eigen::Matrix<float,D,1,Eigen::DontAlign> & val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        #pragma unroll
        for(int i = 0; i < D; i++)
        {
            val[i] += __shfl_down(val[i], offset);
        }
    }
}

template<int D>
__inline__  __device__ void blockReduceSum(Eigen::Matrix<float,D,1,Eigen::DontAlign> & val)
{
    //Allocate shared memory in two steps otherwise NVCC complains about Eigen's non-empty constructor
    static __shared__ unsigned char sharedMem[32 * sizeof(Eigen::Matrix<float,D,1,Eigen::DontAlign>)];

    Eigen::Matrix<float,D,1,Eigen::DontAlign> (&shared)[32] = reinterpret_cast<Eigen::Matrix<float,D,1,Eigen::DontAlign>(&)[32]>(sharedMem);

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : Eigen::Matrix<float,D,1,Eigen::DontAlign>::Zero();

    if(wid == 0)
    {
        warpReduceSum(val);
    }
}

template<int D>
__global__ void reduceSum(Eigen::Matrix<float,D,1,Eigen::DontAlign> * in, Eigen::Matrix<float,D,1,Eigen::DontAlign> * out, int N)
{
    Eigen::Matrix<float,D,1,Eigen::DontAlign> sum = Eigen::Matrix<float,D,1,Eigen::DontAlign>::Zero();

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct Reduction
{
    Eigen::Matrix<float, 3, 3, Eigen::DontAlign> R_prev_curr;
    Eigen::Matrix<float, 3, 1, Eigen::DontAlign> t_prev_curr;

    Intr intr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;

    PtrStep<float> vmap_prev;
    PtrStep<float> nmap_prev;

    float dist_thresh;
    float angle_thresh;

    int cols;
    int rows;
    int N;

    Eigen::Matrix<float,29,1,Eigen::DontAlign> * out;

    //And now for some template metaprogramming magic
    template<int outer, int inner, int end>
    struct SquareUpperTriangularProduct
    {
        __device__ __forceinline__ static void apply(Eigen::Matrix<float,29,1,Eigen::DontAlign> & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * outer) + inner - (outer * (outer + 1) / 2)] = rows[outer] * rows[inner];

            SquareUpperTriangularProduct<outer, inner + 1, end>::apply(values, rows);
        }
    };

    //Inner loop base
    template<int outer, int end>
    struct SquareUpperTriangularProduct<outer, end, end>
    {
        __device__ __forceinline__ static void apply(Eigen::Matrix<float,29,1,Eigen::DontAlign> & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * outer) + end - (outer * (outer + 1) / 2)] = rows[outer] * rows[end];

            SquareUpperTriangularProduct<outer + 1, outer + 1, end>::apply(values, rows);
        }
    };

    //Outer loop base
    template<int end>
    struct SquareUpperTriangularProduct<end, end, end>
    {
        __device__ __forceinline__ static void apply(Eigen::Matrix<float,29,1,Eigen::DontAlign> & values, const float (&rows)[end + 1])
        {
            values[((end + 1) * end) + end - (end * (end + 1) / 2)] = rows[end] * rows[end];
        }
    };

    __device__ __forceinline__ void
    operator () () const
    {
        Eigen::Matrix<float,29,1,Eigen::DontAlign> sum = Eigen::Matrix<float,29,1,Eigen::DontAlign>::Zero();

        SquareUpperTriangularProduct<0, 0, 6> sutp;

        Eigen::Matrix<float,29,1,Eigen::DontAlign> values;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            const int y = i / cols;
            const int x = i - (y * cols);

            const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_curr(vmap_curr.ptr(y)[x],
                                                                   vmap_curr.ptr(y + rows)[x],
                                                                   vmap_curr.ptr(y + 2 * rows)[x]);

            const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_curr_in_prev = R_prev_curr * v_curr + t_prev_curr;

            const Eigen::Matrix<int,2,1,Eigen::DontAlign> p_curr_in_prev(__float2int_rn(v_curr_in_prev(0) * intr.fx / v_curr_in_prev(2) + intr.cx),
                                                                         __float2int_rn(v_curr_in_prev(1) * intr.fy / v_curr_in_prev(2) + intr.cy));

            float row[7] = {0, 0, 0, 0, 0, 0, 0};

            values[28] = 0;

            if(p_curr_in_prev(0) >= 0 && p_curr_in_prev(1) >= 0 && p_curr_in_prev(0) < cols && p_curr_in_prev(1) < rows && v_curr(2) > 0 && v_curr_in_prev(2) > 0)
            {
                const Eigen::Matrix<float,3,1,Eigen::DontAlign> v_prev(__ldg(&vmap_prev.ptr(p_curr_in_prev(1))[p_curr_in_prev(0)]),
                                                                       __ldg(&vmap_prev.ptr(p_curr_in_prev(1) + rows)[p_curr_in_prev(0)]),
                                                                       __ldg(&vmap_prev.ptr(p_curr_in_prev(1) + 2 * rows)[p_curr_in_prev(0)]));

                const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_curr(nmap_curr.ptr(y)[x],
                                                                       nmap_curr.ptr(y + rows)[x],
                                                                       nmap_curr.ptr(y + 2 * rows)[x]);

                const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_curr_in_prev = R_prev_curr * n_curr;

                const Eigen::Matrix<float,3,1,Eigen::DontAlign> n_prev(__ldg(&nmap_prev.ptr(p_curr_in_prev(1))[p_curr_in_prev(0)]),
                                                                       __ldg(&nmap_prev.ptr(p_curr_in_prev(1) + rows)[p_curr_in_prev(0)]),
                                                                       __ldg(&nmap_prev.ptr(p_curr_in_prev(1) + 2 * rows)[p_curr_in_prev(0)]));

                if(n_curr_in_prev.cross(n_prev).norm() < angle_thresh && (v_prev - v_curr_in_prev).norm() < dist_thresh && !isnan(n_curr(0)) && !isnan(n_prev(0)))
                {
                    *(Eigen::Matrix<float,3,1,Eigen::DontAlign>*)&row[0] = n_prev;
                    *(Eigen::Matrix<float,3,1,Eigen::DontAlign>*)&row[3] = v_curr_in_prev.cross(n_prev);
                    row[6] = n_prev.dot(v_prev - v_curr_in_prev);

                    values[28] = 1;

                    sutp.apply(values, row);

                    sum += values;
                }
            }

        }

        blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void estimateKernel(const Reduction reduction)
{
    reduction();
}

void estimateStep(const Eigen::Matrix<float,3,3,Eigen::DontAlign> & R_prev_curr,
                  const Eigen::Matrix<float,3,1,Eigen::DontAlign> & t_prev_curr,
                  const DeviceArray2D<float>& vmap_curr,
                  const DeviceArray2D<float>& nmap_curr,
                  const Intr& intr,
                  const DeviceArray2D<float>& vmap_prev,
                  const DeviceArray2D<float>& nmap_prev,
                  float dist_thresh,
                  float angle_thresh,
                  DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> & sum,
                  DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> & out,
                  float * matrixA_host,
                  float * vectorB_host,
                  float * residual_inliers,
                  int threads, int blocks)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

    Reduction reduction;

    reduction.R_prev_curr = R_prev_curr;
    reduction.t_prev_curr = t_prev_curr;

    reduction.vmap_curr = vmap_curr;
    reduction.nmap_curr = nmap_curr;

    reduction.intr = intr;

    reduction.vmap_prev = vmap_prev;
    reduction.nmap_prev = nmap_prev;

    reduction.dist_thresh = dist_thresh;
    reduction.angle_thresh = angle_thresh;

    reduction.cols = cols;
    reduction.rows = rows;

    reduction.N = cols * rows;
    reduction.out = sum;

    estimateKernel<<<blocks, threads>>>(reduction);

    reduceSum<29><<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[29];
    out.download((Eigen::Matrix<float,29,1,Eigen::DontAlign> *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)  //rows
    {
        for (int j = i; j < 7; ++j)    // cols + b
        {
            float value = host_data[shift++];
            if (j == 6)       // vector b
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

    residual_inliers[0] = host_data[27];
    residual_inliers[1] = host_data[28];
}
