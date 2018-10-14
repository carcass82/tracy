/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

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

__device__ curandState curand_ctx;

__device__ static float fastrand()
{
    return curand_uniform(&curand_ctx);
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

enum { eLAMBERTIAN, eMETAL, eDIELECTRIC, eEMISSIVE };
struct DMaterial
{
    int type;
    float3 albedo;
    float roughness;
    float ior;

    __device__ bool scatter(const DRay& ray, const DIntersection& hit, float3& attenuation, float3& emission, DRay& scattered)
    {
        if (type == eLAMBERTIAN)
        {
            float3 target = hit.point + hit.normal + random_in_unit_sphere();
            scattered.origin = hit.point;
            scattered.direction = target - hit.point;
            attenuation = albedo;
            emission = make_float3(.0f, .0f, .0f);

            return true;
        }
        else if (type == eMETAL)
        {
            float3 reflected = reflect(normalize(ray.direction), hit.normal);
            scattered.origin = hit.point;
            scattered.direction = reflected + roughness * random_in_unit_sphere();
            attenuation = albedo;
            emission = make_float3(.0f, .0f, .0f);
            
            return (dot(scattered.direction, hit.normal) > .0f);
        }
        else if (type == eDIELECTRIC)
        {
            float3 outward_normal;
            attenuation = { 1.f, 1.f, 1.f };
            emission = make_float3(.0f, .0f, .0f);

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
        else if (type == eEMISSIVE)
        {
            emission = albedo;
            return false;
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

struct DBox
{
    float3 min_limit;
    float3 max_limit;
    float3 rot;
    DMaterial material;

    //
    // code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
    //
    __device__ bool intersects(const DRay& ray, DIntersection& hit)
    {
        float3 tmin = (min_limit - ray.origin) / ray.direction;
        float3 tmax = (max_limit - ray.origin) / ray.direction;

        float3 real_min = min(tmin, tmax);
        float3 real_max = max(tmin, tmax);

        float minmax = min(min(real_max.x, real_max.y), real_max.z);
        float maxmin = max(max(real_min.x, real_min.y), real_min.z);

        if (minmax >= maxmin && maxmin < hit.t)
        {
            hit.t = maxmin;
            hit.point = ray.point_at(maxmin);
            hit.normal = normal(hit.point);
            hit.material = &material;
            return true;
        }
        
        return false;
    }

    __device__ float3 normal(const float3& point)
    {
        const float eps = .001f;

        if (fabs(min_limit.x - point.x) < eps) return make_float3(-1.f,  .0f,  .0f);
        if (fabs(max_limit.x - point.x) < eps) return make_float3( 1.f,  .0f,  .0f);
        if (fabs(min_limit.y - point.y) < eps) return make_float3( .0f, -1.f,  .0f);
        if (fabs(max_limit.y - point.y) < eps) return make_float3( .0f,  1.f,  .0f);
        if (fabs(min_limit.z - point.z) < eps) return make_float3( .0f,  .0f, -1.f);
        return make_float3(.0f, .0f, 1.f);
    }
};

const int MAX_DEPTH = 5;
template<int depth>
__device__ float3 get_color_for(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count)
{
    //
    // check for hits
    //
    bool hit = false;
    DIntersection hit_data;
    hit_data.t = FLT_MAX;
    for (int i = 0; i < sphere_count; ++i)
    {
        DSphere& sphere = spheres[i];
        if (sphere.intersects(ray, hit_data))
        {
            hit = true;
        }
    }

    for (int i = 0; i < box_count; ++i)
    {
        DBox& box = boxes[i];
        if (box.intersects(ray, hit_data))
        {
            hit = true;
        }
    }
    
    //
    // return color or continue
    //
    if (hit)
    {
        //
        // debug - show normals
        //
        //return .5f * (1.f + normalize(hit_data.normal));

        DRay scattered;
        float3 attenuation;
        float3 emission;
        if (hit_data.material && hit_data.material->scatter(ray, hit_data, attenuation, emission, scattered))
        {
            return emission + attenuation * get_color_for<depth + 1>(scattered, spheres, sphere_count, boxes, box_count);
        }
        else
        {
            return emission;
        }
    }

    //return make_float3(1.f, 1.f, 1.f);
    return make_float3(.0f, .0f, 0.0f);
}

template<>
__device__ float3 get_color_for<MAX_DEPTH>(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count)
{
    //return make_float3(1.f, 1.f, 1.f);
    return make_float3(.0f, .0f, .0f);
}

__global__ void raytrace(int width, int height, int samples, float3* pixels)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    curand_init(j, i, 0, &curand_ctx);

    //
    // scene definition
    //
    DSphere spheres[3];
    int spherecount = array_size(spheres);
    spheres[0].center = { 130 + 82.5 - 25, 215, 65 + 82.5 - 25 };
    spheres[0].radius = 50.f;
    spheres[0].material.type = eDIELECTRIC;
    spheres[0].material.ior = 1.5f;

    spheres[1].center = { 265 + 82.5 + 35, 400, 295 + 82.5 - 35 };
    spheres[1].radius = 70.f;
    spheres[1].material.type = eMETAL;
    spheres[1].material.albedo = { .8f, .85f, .88f };
    spheres[1].material.roughness = .05f;
    
    spheres[2].center = { 265 + 82.5 + 15, 30, 80 };
    spheres[2].radius = 30.f;
    spheres[2].material.type = eMETAL;
    spheres[2].material.albedo = { 1.f, .71f, .29f };
    spheres[2].material.roughness = .05f;


    DBox boxes[8];
    int boxcount = array_size(boxes);
    // light
    boxes[0].min_limit = { 213.f, 554.f, 227.f };
    boxes[0].max_limit = { 343.f, 555.f, 332.f };
    boxes[0].rot = { .0f, .0f, .0f };
    boxes[0].material.type = eEMISSIVE;
    boxes[0].material.albedo = { 15.f, 15.f, 15.f };
    // green side
    boxes[1].min_limit = { 555.f,    .0f, 0.f};
    boxes[1].max_limit = { 555.1f, 555.f, 555.f};
    boxes[1].rot = { .0f, .0f, .0f };
    boxes[1].material.type = eLAMBERTIAN;
    boxes[1].material.albedo = { 0.12f, 0.45f, .15f };
    // red side
    boxes[2].min_limit = { -0.1f,   .0f, 0.f };
    boxes[2].max_limit = {  .0f, 555.f, 555.f };
    boxes[2].rot = { .0f, .0f, .0f };
    boxes[2].material.type = eLAMBERTIAN;
    boxes[2].material.roughness = .0f;
    boxes[2].material.albedo = { 0.65f, .05f, .05f };
    // floor
    boxes[3].min_limit = { .0f,    -.1f, 0.f };
    boxes[3].max_limit = { 555.f, 0.f, 555.f };
    boxes[3].rot = { .0f, .0f, .0f };
    boxes[3].material.type = eLAMBERTIAN;
    boxes[3].material.albedo = { 0.73f, .73f, .73f };
    // roof
    boxes[4].min_limit = { .0f,    555.f, 0.f };
    boxes[4].max_limit = { 555.f, 555.1f, 555.f };
    boxes[4].rot = { .0f, .0f, .0f };
    boxes[4].material.type = eLAMBERTIAN;
    boxes[4].material.albedo = { 0.73f, .73f, .73f };
    //back
    boxes[5].min_limit = { .0f,    .0f, 554.9f };
    boxes[5].max_limit = { 555.f, 555.f, 555.f };
    boxes[5].rot = { .0f, .0f, .0f };
    boxes[5].material.type = eLAMBERTIAN;
    boxes[5].material.albedo = { 0.73f, .73f, .73f };
    // higher block
    boxes[6].min_limit = { 265.f,   .0f, 295.f };
    boxes[6].max_limit = { 430.f, 330.f, 460.f };
    boxes[6].rot = { .0f, 15.0f, .0f };
    boxes[6].material.type = eLAMBERTIAN;
    boxes[6].material.albedo = { 0.73f, .73f, .73f };
    // lower block
    boxes[7].min_limit = { 130.f,   .0f, 65.f };
    boxes[7].max_limit = { 295.f, 165.f, 230.f };
    boxes[7].rot = { .0f, -18.f, .0f };
    boxes[7].material.type = eLAMBERTIAN;
    boxes[7].material.albedo = { 0.73f, .73f, .73f };

    //
    // camera setup
    //
    const float fov = radians(40.f);
    const float aspect = width / fmax(1.f, static_cast<float>(height));
    const float3 center{ 278.f, 278.f, -800.f };
    const float3 lookat{ 278.f, 278.f, 0.f };
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

        color += get_color_for<0>(ray, spheres, spherecount, boxes, boxcount);
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
