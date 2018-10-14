/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>

#include "mathutils.cuh"


#ifdef _DEBUG
#define CUDALOG(...) printf(__VA_ARGS__)
#else 
#define CUDALOG(...) do {} while(0);
#endif

//
// from helper_cuda.h
// NVidia CUDA samples
// 
template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        cudaError_t cuda_error = cudaGetLastError();

        CUDALOG("[CUDA error] at %s:%d code=%d (%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(cuda_error), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

//
// ----------------------------------------------------------------------------
//

__device__ float schlick_fresnel(float costheta, float ior)
{
    float r0 = (1.f - ior) / (1.f + ior);
    r0 *= r0;
    return r0 + (1.f - r0) * powf((1.f - costheta), 5);
}

__device__ static float fastrand()
{
    static int s_seed = 123456789;

    s_seed = (214013 * s_seed + 2531011);
    return ((s_seed >> 16) & 0x7FFF) / 32768.0f;
}

__device__ float3 random_in_unit_sphere()
{
    float3 p;
    do
    {
        p = make_float3(2.f * fastrand() - 1.f, 2.f * fastrand() - 1.f, 2.f * fastrand() - 1.f);
    } while (dot(p, p) >= 1.f);

    return p;
}

//
// ----------------------------------------------------------------------------
//

struct DRay
{
    float3 origin;
    float3 direction;

    __device__ float3 point_at(float t) const { return origin + t * direction; }
};

struct DMaterial;
struct DIntersection
{
    float t;
    float3 point;
    float3 normal;
    DMaterial* material;
};

enum { eLAMBERTIAN, eMETAL, eDIELECTRIC };
struct DMaterial
{
    int type;
    float3 albedo;
    float roughness;
    float ior;

    __device__ bool scatter(const DRay& ray, const DIntersection& hit, float3& attenuation, DRay& scattered)
    {
        if (type == eLAMBERTIAN)
        {
            float r1 = 2.f * PI * fastrand();
            float r2 = fastrand();
            float r2s = sqrtf(r2);

            float3 w = hit.normal;
            float3 u = normalize(cross(fabs(w.x) > 1.f ? float3{ 0,1,0 } : float3{ 1,0,0 }, w));
            float3 v = cross(w, u);

            scattered.origin = hit.point;
            scattered.direction = normalize(r2s * cosf(r1) * u + r2s * sinf(r1) * v + sqrtf(1 - r2) * w);

            attenuation = albedo;

            return true;
        }
        else if (type == eMETAL)
        {
            float3 reflected = reflect(normalize(ray.direction), hit.normal);
            scattered.origin = hit.point;
            scattered.direction = reflected /*+ roughness * random_in_unit_sphere()*/;
            attenuation = albedo;
            
            return (dot(scattered.direction, hit.normal) > .0f);
        }
        else if (type == eDIELECTRIC)
        {
            float3 outward_normal;
            attenuation = { 1.f, 1.f, 1.f };

            float ni_nt;
            float cosine;
            if (dot(ray.direction, hit.normal) > .0f)
            {
                outward_normal = -1.f * hit.normal;
                ni_nt = ior;
                cosine = ior * dot(ray.direction, hit.normal) / length(ray.direction);
            }
            else
            {
                outward_normal = hit.normal;
                ni_nt = 1.f / ior;
                cosine = (dot(ray.direction, hit.normal) * -1.f) / length(ray.direction);
            }

            float3 refracted;
            bool is_refracted = refract(ray.direction, normalize(outward_normal), ni_nt, refracted);
            float reflect_chance = (is_refracted) ? schlick_fresnel(cosine, ior) : 1.0f;
            
            scattered.origin = hit.point;
            scattered.direction = (fastrand() < reflect_chance) ? reflect(normalize(ray.direction), hit.normal) : refracted;

            return true;
        }

        return false;
    }
};

struct DSphere
{
    float3 center;
    float radius;
    DMaterial material;

    __device__ bool intersects(const DRay& ray, DIntersection& hit)
    {
        float3 oc = ray.origin - center;
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        
        if (b <= .0f || c <= .0f)
        {
            float discriminant = b * b - c;
            if (discriminant > 0.f)
            {
                discriminant = sqrtf(discriminant);
                float t0 = -b - discriminant;
                float t1 = -b + discriminant;
                if (t0 > .0001f && t0 < hit.t)
                {
                    hit.t = t0;
                    hit.point = ray.point_at(t0);
                    hit.normal = (hit.point - center) / radius;
                    hit.material = &material;
                    return true;
                }
                if (t1 > .0001f && t1 < hit.t)
                {
                    hit.t = t1;
                    hit.point = ray.point_at(t1);
                    hit.normal = (hit.point - center) / radius;
                    hit.material = &material;
                    return true;
                }
            }
        }
        return false;
    }
};

const int MAX_DEPTH = 5;
template<int depth>
__device__ float3 get_color_for(DRay ray, DSphere* world, int sphere_count)
{
    //
    // check for hits
    //
    bool hit = false;
    DIntersection hit_data;
    hit_data.t = FLT_MAX;
    for (int i = 0; i < sphere_count; ++i)
    {
        DSphere& sphere = world[i];
        if (sphere.intersects(ray, hit_data))
        {
            hit = true;
        }
    }
    
    //
    // return color or continue
    //
    if (hit)
    {
        DRay scattered;
        float3 attenuation;
        if (hit_data.material && hit_data.material->scatter(ray, hit_data, attenuation, scattered))
        {
            return attenuation * get_color_for<depth + 1>(scattered, world, sphere_count);
        }
    }

    float3 direction = normalize(ray.direction);
    float t = .5f * (direction.y + 1.0f);

    return (1.f - t) * float3 { 1.f, 1.f, 1.f } +t * float3{ .5f, .7f, 1.f };
}

template<>
__device__ float3 get_color_for<MAX_DEPTH>(DRay ray, DSphere* world, int sphere_count)
{
    float3 direction = normalize(ray.direction);
    float t = .5f * (direction.y + 1.0f);

    return (1.f - t) * float3 { 1.f, 1.f, 1.f } +t * float3{ .5f, .7f, 1.f };
}

__global__ void raytrace(int width, int height, int samples, float3* pixels)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    //
    // scene definition
    //
    DSphere spheres[4];
    spheres[0].center = { 0.f, 0.f, -1.f };
    spheres[0].radius = 0.5;
    spheres[0].material.type = eLAMBERTIAN;
    spheres[0].material.albedo = { 0.1f, 0.2f, 0.5f };

    spheres[1].center = { 0.f, -100.5f, -1.f };
    spheres[1].radius = 100.f;
    spheres[1].material.type = eLAMBERTIAN;
    spheres[1].material.albedo = { 0.8f, 0.8f, 0.f };

    spheres[2].center = { 1.f, 0.f, -1.f };
    spheres[2].radius = 0.5f;
    spheres[2].material.type = eMETAL;
    spheres[2].material.albedo = { 0.8f, 0.6f, 0.2f };
    spheres[2].material.roughness = .05f;
    
    spheres[3].center = { -1.f, 0.f, -1.f };
    spheres[3].radius = .5f;
    spheres[3].material.type = eDIELECTRIC;
    spheres[3].material.ior = 1.5f;

    //
    // camera setup
    //
    const float fov = radians(45.f);
    const float aspect = width / fmax(1.f, static_cast<float>(height));
    const float3 center{ -.5f, 1.2f, 1.5f };
    const float3 lookat{ 0.f, 0.f, -1.f };
    const float3 vup{ 0.f, 1.f, 0.f };

    float height_2 = tanf(fov / 2.f);
    float width_2 = height_2 * aspect;

    float3 w = normalize(center - lookat);
    float3 u = normalize(cross(vup, w));
    float3 v = cross(w, u);

    float3 horizontal = 2.f * width_2 * u;
    float3 vertical = 2.f * height_2 * v;
    float3 origin = center - width_2 * u - height_2 * v - w;
    
    //
    // main loop
    //
    float3 color{ .0f, .0f, .0f };
    for (int sample = 0; sample < samples; ++sample)
    {
        float s = ((i + fastrand()) / static_cast<float>(width));
        float t = ((j + fastrand()) / static_cast<float>(height));

        DRay ray;
        ray.origin = center;
        ray.direction = normalize(origin + s * horizontal + t * vertical - center);

        color += get_color_for<0>(ray, spheres, array_size(spheres));
    }
    
    //
    // debug output if needed
    //
    //color.x = s;
    //color.y = t;
    //color.z = .0f;
    
    float3& pixel = *(&pixels[j * width + i]);
    if (i < width && j < height)
    {
        pixel.x = color.x;
        pixel.y = color.y;
        pixel.z = color.z;
    }

    // just to be sure we're running
    if (i == 0 && j == 0) { CUDALOG("[CUDA] running kernel...\n"); }
}

//
// IFace for raytracer.cpp
// 
extern "C" void cuda_trace(int w, int h, int s, float* out_buffer)
{
    const int MAX_GPU = 32;
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < num_gpus; i++)
    {
        cudaGetDeviceProperties(&gpu_properties[i], i);

        CUDALOG("Device %d (%s):\n\t%d threads\n\tblocksize: %dx%dx%d\n\tshmem per block: %lu Kb\n\tgridsize: %dx%dx%d\n\n",
               i,
               gpu_properties[i].name,
               gpu_properties[i].maxThreadsPerBlock,
               gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
               static_cast<unsigned long>(gpu_properties[i].sharedMemPerBlock / 1024.f),
               gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);
    }

    CUDALOG("image is %dx%d (%d samples desired)\n", w, h, s);

    //
    // TODO: resolve crash with multi gpu
    //
    num_gpus = 1;
    //
    // TODO: resolve crash with multi gpu
    //

    const int new_w = w / num_gpus;
    const int new_h = h;

    float3* d_output_cuda[MAX_GPU];

    for (int i = 0; i < num_gpus; ++i)
    {
        cudaSetDevice(i);

        checkCudaErrors(cudaMalloc((void**)&d_output_cuda[i], new_w * new_h * 3 * sizeof(float)));

        int threads_per_row = sqrt(gpu_properties[i].maxThreadsPerBlock) / 2;
        dim3 dimBlock(threads_per_row, threads_per_row);
        dim3 dimGrid(w / dimBlock.x + 1, h / dimBlock.y + 1);
        
        raytrace<<<dimGrid, dimBlock>>>(new_w, new_h, s, d_output_cuda[i]);
        CUDALOG("raytrace<<<(%d,%d,%d), (%d,%d,%d)>>>\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
    }

    for (int i = 0; i < num_gpus; ++i)
    {
        cudaSetDevice(i);
        checkCudaErrors(cudaDeviceSynchronize());

        CUDALOG("cuda compute (%d/%d) completed!\n", i, num_gpus - 1);

        checkCudaErrors(cudaMemcpy(out_buffer + (i * new_w * new_h * 3 * sizeof(float)), d_output_cuda[i], new_w * new_h * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_output_cuda[i]));
    }
}
