/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include <cuda_runtime.h>
#include "log.h"

#define CUDAAssert(val) cuda::ensure((val), __FILE__, __LINE__)

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
        TracyLog("[CUDA Error] at %s:%d code=%d (%s)\n", file, line, static_cast<uint32_t>(val), cudaGetErrorName(val));
        cudaDeviceReset();

        DEBUG_BREAK();

#if defined(NDEBUG)
        exit(EXIT_FAILURE);
#endif
    }
}
}
