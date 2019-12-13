#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"
#include "log.h"

#include "ray.h"
#include "camera.h"
#include "material.h"

constexpr int MAX_GPU = 32;
constexpr int MAX_DEPTH = 5;

__device__ constexpr inline uint32_t ToInt(vec3 color)
{
    color.r = clamp(color.r, 0.0f, 255.0f);
    color.g = clamp(color.g, 0.0f, 255.0f);
    color.b = clamp(color.b, 0.0f, 255.0f);

    uint32_t packed = (uint8_t(color.b) << 16) | (uint8_t(color.g) << 8) | uint8_t(color.r);
    return packed;
}

__device__ inline float fastrand(curandState* curand_ctx)
{
    return curand_normal(curand_ctx);
}

__device__ inline vec3 TraceInternal(const Ray& in_ray)
{
    vec3 current_color = { 1.f, 1.f, 1.f };
    Ray current_ray = { in_ray };

    HitData hit_data;
    hit_data.t = FLT_MAX;

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        if (true) //if (details_->ComputeIntersection(*scene_, current_ray, intersection_data))
        {

#if DEBUG_SHOW_NORMALS
            //return .5f * normalize((1.f + mat3(camera_->GetView()) * hit_data.normal));
#else
            Ray scattered;
            vec3 attenuation;
            vec3 emission;
            if (hit_data.material->Scatter(current_ray, hit_data, attenuation, emission, scattered))
            {
                current_color *= attenuation;
                current_ray = scattered;
            }
            //else
            {
                current_color *= emission;
                return current_color;
            }
#endif
        }
        else
        {
            return {};
        }
    }

    return {};
}

__global__ void Trace(uint32_t* output, int width, int height)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    curandState curand_ctx;
    curand_init(clock64(), i, j, &curand_ctx);

    //
    // main loop
    //
    
    //int raycount_inst = 0;

    float f_width = static_cast<float>(width);
    float f_height = static_cast<float>(height);

    Camera c;
    c.Setup({ 0.5, 0.5, 1.5 }, { 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, 60.f, f_width / max(1.f, f_height));

    vec3 color{};
    for (int sample = 0; sample < 1; ++sample)
    {
        float s = ((i + fastrand(&curand_ctx)) / f_width);
        float t = ((j + fastrand(&curand_ctx)) / f_height);

        Ray r = c.GetRayFrom(s, t);
        color += TraceInternal(r);
    }
    //atomicAdd(raycount, raycount_inst);

    output[j * width + i] = ToInt(color * 255.99f);
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
    Trace<<<grid, block>>>(output, w, h);

    CUDAAssert(cudaGetLastError());
}
