/*
 * safe_call.hpp
 *
 *  Created on: 1 Sep 2014
 *      Author: thomas
 */

#ifndef SAFE_CALL_HPP_
#define SAFE_CALL_HPP_

#include <cuda_runtime_api.h>
#include <cstdlib>
#include "initialization.hpp"

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

#endif /* SAFE_CALL_HPP_ */
