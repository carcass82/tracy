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

namespace cuda
{
//
// from helper_cuda.h
// NVidia CUDA samples
//
inline void ensure(cudaError_t val, const char* file, int line)
{
    if (val != cudaSuccess)
    {
        CUDALog("[CUDA Error] at %s:%d code=%d (%s)\n", file, line, static_cast<unsigned int>(val), cudaGetErrorName(val));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
}
