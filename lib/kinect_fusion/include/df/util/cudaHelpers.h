#pragma once

#include <stdexcept>
#include <cuda_runtime.h>
#include <device_functions.h>

#define CheckCudaDieOnError() df::_CheckCudaDieOnError( __FILE__, __LINE__ );
namespace df {

inline void _CheckCudaDieOnError( const char * sFile, const int nLine ) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(sFile) + ":" + std::to_string(nLine) + "\nCUDA error: " + cudaGetErrorString(error));
    }
}

template <typename I>
inline I intDivideAndCeil(I numerator, I denominator) {
    return (numerator + denominator - 1) / denominator;
}

inline __host__ __device__ uint maxThreadsPerBlock() {
    return 1024;
}

template <typename Scalar>
__host__ __device__
inline Scalar expX(Scalar);

template <>
__host__ __device__
inline float expX(float val) {

    return expf(val);

}

template <>
__host__ __device__
inline double expX(double val) {

    return exp(val);

}

template <typename Scalar>
inline __device__ Scalar atomicAddX(Scalar * addres, Scalar val) {

    return atomicAdd(addres,val);

}

#ifdef __CUDACC__
template <>
inline __device__ double atomicAddX(double * address, double val) {
    unsigned long long int * address_as_ull =
            (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif // __CUDACC__

} // namespace df


