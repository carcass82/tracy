#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"
#include "log.h"

constexpr int MAX_GPU = 32;

__device__ inline int ColorByteToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ inline float fastrand(curandState* curand_ctx)
{
    return curand_normal(curand_ctx);
}

__global__ void cudaProcess(unsigned int* output, int width, int height)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    curandState curand_ctx;
    curand_init(clock64(), i, j, &curand_ctx);

    float s = fastrand(&curand_ctx) * 255.99;

    output[j * width + i] = ColorByteToInt(s, s, s);
}

extern "C" void cuda_setup()
{
    int num_gpus = 0;
    CUDAAssert(cudaGetDeviceCount(&num_gpus));
    
    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < num_gpus; i++)
    {
        CUDAAssert(cudaGetDeviceProperties(&gpu_properties[i], i));

        CUDALog("Device %d (%s):\n"
                "\t%d threads\n"
                "\tblocksize: %dx%dx%d\n"
                "\tshmem per block: %dKb\n"
                "\tgridsize: %dx%dx%d\n\n",
                i,
                gpu_properties[i].name,
                gpu_properties[i].maxThreadsPerBlock,
                gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
                gpu_properties[i].sharedMemPerBlock / 1024,
                gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);
    }

    CUDAAssert(cudaSetDevice(0));
}

extern "C" void cuda_trace(unsigned int* output, int w, int h)
{
    CUDAAssert(cudaSetDevice(0));

    dim3 block(16, 16, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    cudaProcess<<<grid, block>>>(output, w, h);

    CUDAAssert(cudaGetLastError());
}
