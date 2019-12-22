/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <cuda_runtime.h>
#include "log.h"

#define CUDALog(msg, ...) TracyLog(msg, __VA_ARGS__)
#define CUDAAssert(val)   cuda::ensure((val), __FILE__, __LINE__)
#define CUDACheck(val)    cuda::check((val), #val, __FILE__, __LINE__)

namespace cuda
{
//
// from helper_cuda.h
// NVidia CUDA samples
// 
template <typename T>
inline void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        cudaError_t cuda_error = cudaGetLastError();

        CUDALog("[CUDA Error] at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(cuda_error), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


inline void ensure(cudaError_t val, const char* const file, int const line)
{
    if (val != cudaSuccess)
    {
        CUDALog("[CUDA Error] at %s:%d code=%d (%s)\n", file, line, static_cast<unsigned int>(val), cudaGetErrorName(val));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
}
