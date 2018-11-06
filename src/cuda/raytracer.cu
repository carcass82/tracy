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


#ifndef NODEBUG
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
    return r0 + (1.f - r0) * powf(max(.0f, (1.f - costheta)), 5);
}

__device__ curandState curand_ctx;
__device__ float cuda_fastrand()
{
    return curand_uniform(&curand_ctx);
}

__device__ float3 cuda_random_on_unit_sphere()
{
    float z = cuda_fastrand() * 2.f - 1.f;
    float a = cuda_fastrand() * 2.f * PI;
    float r = sqrtf(max(.0f, 1.f - z * z));

    return make_float3(r * cosf(a), r * sinf(a), z);
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
    int type;
    int index;
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
            float3 target = hit.point + hit.normal + cuda_random_on_unit_sphere();
            scattered.origin = hit.point;
            scattered.direction = normalize(target - hit.point);
            attenuation = albedo;
            emission = make_float3(.0f, .0f, .0f);

            return true;
        }
        else if (type == eMETAL)
        {
            float3 reflected = reflect(normalize(ray.direction), hit.normal);
            scattered.origin = hit.point;
            scattered.direction = reflected + roughness * cuda_random_on_unit_sphere();

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
                cosine = dot(ray.direction, hit.normal) / length(ray.direction);
                cosine = sqrtf(1.f - ior * ior * (1.f - cosine - cosine));
            }
            else
            {
                outward_normal = hit.normal;
                ni_nt = 1.f / ior;
                cosine = -dot(ray.direction, hit.normal) / length(ray.direction);
            }

            float3 refracted;
            bool is_refracted = refract(ray.direction, outward_normal, ni_nt, refracted);
            float reflect_chance = (is_refracted) ? schlick_fresnel(cosine, ior) : 1.0f;
            
            scattered.origin = hit.point;
            scattered.direction = normalize((cuda_fastrand() < reflect_chance) ? reflect(ray.direction, hit.normal) : refracted);

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

    __device__ void hit_data(const DRay& ray, DIntersection& hit)
    {
        hit.point = ray.point_at(hit.t);
        hit.normal = (hit.point - center) / radius;
        hit.material = &material;
    }
};

struct DBox
{
    float3 min_limit;
    float3 max_limit;
    float3 rot;
    DMaterial material;

    __device__ void hit_data(const DRay& ray, DIntersection& hit)
    {
        hit.point = ray.point_at(hit.t);
        hit.normal = normal(hit.point);
        hit.material = &material;
    }

    __device__ float3 normal(const float3& point)
    {
        if (fabs(min_limit.x - point.x) < EPS) return make_float3(-1.f,  .0f,  .0f);
        if (fabs(max_limit.x - point.x) < EPS) return make_float3( 1.f,  .0f,  .0f);
        if (fabs(min_limit.y - point.y) < EPS) return make_float3( .0f, -1.f,  .0f);
        if (fabs(max_limit.y - point.y) < EPS) return make_float3( .0f,  1.f,  .0f);
        if (fabs(min_limit.z - point.z) < EPS) return make_float3( .0f,  .0f, -1.f);
        return make_float3(.0f, .0f, 1.f);
    }
};

enum { eSPHERE, eBOX };
__device__ bool intersect_spheres(DRay ray, DSphere* spheres, int sphere_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < sphere_count; ++i)
    {
        DSphere& sphere = spheres[i];

        float3 oc = ray.origin - sphere.center;
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - sphere.radius * sphere.radius;

        if (b <= .0f || c <= .0f)
        {
            float discriminant = b * b - c;
            if (discriminant > 0.f)
            {
                discriminant = sqrtf(discriminant);

                float t0 = -b - discriminant;
                if (t0 > EPS && t0 < hit_data.t)
                {
                    hit_data.t = t0;
                    hit_data.type = eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }

                float t1 = -b + discriminant;
                if (t1 > EPS && t1 < hit_data.t)
                {
                    hit_data.t = t1;
                    hit_data.type = eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }
            }
        }
    }

    return hit_something;
}

__device__ bool intersect_boxes(DRay ray, DBox* boxes, int box_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < box_count; ++i)
    {
        DBox& box = boxes[i];

        float tmin = EPS;
        float tmax = FLT_MAX;

        bool boxhit = false;
        for (int side = 0; side < 3; ++side)
        {
            // TODO: think something better
            float direction = (side == 0) ? ray.direction.x : (side == 1) ? ray.direction.y : ray.direction.z;
            float origin = (side == 0) ? ray.origin.x : (side == 1) ? ray.origin.y : ray.origin.z;
            float minbound = (side == 0) ? box.min_limit.x : (side == 1) ? box.min_limit.y : box.min_limit.z;
            float maxbound = (side == 0) ? box.max_limit.x : (side == 1) ? box.max_limit.y : box.max_limit.z;

            if (fabs(direction) < EPS)
            {
                if (origin < minbound || origin > maxbound) { boxhit = false; break; }
            }
            else
            {
                float ood = 1.f / direction;
                float t1 = (minbound - origin) * ood;
                float t2 = (maxbound - origin) * ood;

                if (t1 > t2) swap(t1, t2);

                tmin = max(tmin, t1);
                tmax = min(tmax, t2);

                if (tmin > tmax || tmin > hit_data.t) { boxhit = false; break; }
                boxhit = true;
            }
        }

        if (boxhit)
        {
            hit_data.t = tmin;
            hit_data.type = eBOX;
            hit_data.index = i;
            hit_something = true;
        }
    }

    return hit_something;
}

__device__ const int MAX_DEPTH = 5;
__device__ const float3 WHITE = {1.f, 1.f, 1.f};
__device__ const float3 BLACK = {0.f, 0.f, 0.f};

template<int depth>
__device__ float3 get_color_for(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count, size_t* raycount)
{
    //
    // check for hits
    //
    bool hitspheres = false;
    bool hitboxes = false;
    DIntersection hit_data;
    hit_data.t = FLT_MAX;

    hitspheres = intersect_spheres(ray, spheres, sphere_count, hit_data);
    hitboxes = intersect_boxes(ray, boxes, box_count, hit_data);

    ++(*raycount);

    //
    // return color or continue
    //
    if (hitspheres || hitboxes)
    {
        //
        // debug - show normals
        //
        //return .5f * (1.f + normalize(hit_data.normal));

        if (hit_data.type == eSPHERE)
        {
            spheres[hit_data.index].hit_data(ray, hit_data);
        }
        else
        {
            boxes[hit_data.index].hit_data(ray, hit_data);
        }

        DRay scattered;
        float3 attenuation;
        float3 emission;
        if (hit_data.material && hit_data.material->scatter(ray, hit_data, attenuation, emission, scattered))
        {
            return emission + attenuation * get_color_for<depth + 1>(scattered, spheres, sphere_count, boxes, box_count, raycount);
        }
        else
        {
            return emission;
        }
    }

    //return WHITE;
    return BLACK;
}

template<>
__device__ float3 get_color_for<MAX_DEPTH>(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count, size_t* raycount)
{
    //return WHITE;
    return BLACK;
}

__global__ void raytrace(int width, int height, int samples, float3* pixels, size_t* raycount)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    curand_init(clock64(), i, 0, &curand_ctx);

    //
    // scene definition
    //
#define CORNELL 0
#if CORNELL
    __shared__ DSphere spheres[3];
    int spherecount = array_size(spheres);
    spheres[0].center = { 130 + 82.5 - 25, 215, 65 + 82.5 - 25 };
    spheres[0].radius = 50.f;
    spheres[0].material.type = eDIELECTRIC;
    spheres[0].material.ior = 1.5f;
    //spheres[0].material.type = eEMISSIVE;
    //spheres[0].material.albedo = { 15.f, 15.f, 15.f };

    spheres[1].center = { 265 + 82.5 + 35, 400, 295 + 82.5 - 35 };
    spheres[1].radius = 70.f;
    spheres[1].material.type = eMETAL;
    spheres[1].material.albedo = { .8f, .85f, .88f };
    spheres[1].material.roughness = .0f;
    //spheres[1].material.type = eEMISSIVE;
    //spheres[1].material.albedo = { 15.f, 15.f, 15.f };
    
    spheres[2].center = { 265 + 82.5 + 15, 30, 80 };
    spheres[2].radius = 30.f;
    spheres[2].material.type = eMETAL;
    spheres[2].material.albedo = { 1.f, .71f, .29f };
    spheres[2].material.roughness = .05f;
    //spheres[2].material.type = eEMISSIVE;
    //spheres[2].material.albedo = { 15.f, 15.f, 15.f };


    __shared__ DBox boxes[8];
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
    //boxes[4].material.type = eEMISSIVE;
    //boxes[4].material.albedo = { 1.f, 1.f, 1.f };
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

#else
    __shared__ DSphere spheres[8];
    int spherecount = array_size(spheres);
    spheres[0].center = { 0.f, 0.f, -1.f };
    spheres[0].radius = .5f;
    spheres[0].material.type = eLAMBERTIAN;
    spheres[0].material.albedo = { 0.1f, 0.2f, 0.5f };

    spheres[1].center = { 0.f, -100.5f, -1.f };
    spheres[1].radius = 100.f;
    spheres[1].material.type = eLAMBERTIAN;
    spheres[1].material.albedo = { 0.2f, 0.2f, 0.2f };

    spheres[2].center = { 1.f, 0.f, -1.f };
    spheres[2].radius = .5f;
    spheres[2].material.type = eMETAL;
    spheres[2].material.albedo = { .8f, .85f, .88f };
    spheres[2].material.roughness = .0f;

    spheres[3].center = { -1.f, 0.f, -1.f };
    spheres[3].radius = .5f;
    spheres[3].material.type = eDIELECTRIC;
    spheres[3].material.ior = 1.5f;
    //spheres[3].material.type = eMETAL;
    //spheres[3].material.albedo = { .8f, .85f, .88f };
    //spheres[3].material.roughness = .05f;

    spheres[4].center = { 0.f, 150.f, -1.f };
    spheres[4].radius = 100.f;
    spheres[4].material.type = eEMISSIVE;
    spheres[4].material.albedo = { 2.f, 2.f, 2.f };

    spheres[5].center = { 0.f, 0.f, 0.f };
    spheres[5].radius = .2f;
    //spheres[5].material.type = eEMISSIVE;
    //spheres[5].material.albedo = { 2.f, 2.f, 2.f };
    spheres[5].material.type = eDIELECTRIC;
    spheres[5].material.ior = 1.5f;

    spheres[6].center = { 0.f, 1.f, -1.5f };
    spheres[6].radius = .3f;
    spheres[6].material.type = eMETAL;
    spheres[6].material.albedo = { 1.f, .71f, .29f };
    spheres[6].material.roughness = .05f;

    spheres[7].center = { 0.f, 0.f, -2.5f };
    spheres[7].radius = .5f;
    spheres[7].material.type = eLAMBERTIAN;
    spheres[7].material.albedo = { .85f, .05f, .02f };

    __shared__ DBox boxes[1];
    int boxcount = array_size(boxes);
    boxes[0].min_limit = { -2.f, 0.f, -3.1f };
    boxes[0].max_limit = { 2.f, 2.f, -3.f };
    boxes[0].material.type = eLAMBERTIAN;
    boxes[0].material.albedo = { .05f, .85f, .02f };

    //
    // camera setup
    //
    const float fov = radians(60.f);
    const float aspect = width / fmax(1.f, static_cast<float>(height));
    const float3 center{ -.5f, 1.2f, 1.5f };
    const float3 lookat{ 0.f, 0.f, -1.f };
    const float3 vup{ 0.f, 1.f, 0.f };
#endif

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
    size_t raycount_inst = 0;
    float3 color{ .0f, .0f, .0f };
    for (int sample = 0; sample < samples; ++sample)
    {
        float s = ((i + cuda_fastrand()) / static_cast<float>(width));
        float t = ((j + cuda_fastrand()) / static_cast<float>(height));

        DRay ray;
        ray.origin = center;
        ray.direction = normalize(origin + s * horizontal + t * vertical - center);

        color += get_color_for<0>(ray, spheres, spherecount, boxes, boxcount, &raycount_inst);
    }

    atomicAdd(raycount, raycount_inst);
    
    //
    // debug output if needed
    //
    //color.x = s;
    //color.y = t;
    //color.z = .0f;
    
    float3& pixel = *(&pixels[j * width + i]);
    //pixel.x = color.x;
    //pixel.y = color.y;
    //pixel.z = color.z;

    atomicAdd(&pixel.x, color.x);
    atomicAdd(&pixel.y, color.y);
    atomicAdd(&pixel.z, color.z);

    // just to be sure we're running
    if (i == 0 && j == 0 && blockIdx.z == 0) { CUDALOG("[CUDA] running kernel...\n"); }
}

//
// IFace for raytracer.cpp
// 
extern "C" void cuda_trace(int w, int h, int ns, float* out_buffer, size_t& raycount)
{
    // ensure output buffer is properly zeroed
    memset(out_buffer, 0, w * h * sizeof(float3));

    const int MAX_GPU = 32;
    int num_gpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));

    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < num_gpus; i++)
    {
        checkCudaErrors(cudaGetDeviceProperties(&gpu_properties[i], i));

        CUDALOG("Device %d (%s):\n\t%d threads\n\tblocksize: %dx%dx%d\n\tshmem per block: %lu Kb\n\tgridsize: %dx%dx%d\n\n",
               i,
               gpu_properties[i].name,
               gpu_properties[i].maxThreadsPerBlock,
               gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
               static_cast<unsigned long>(gpu_properties[i].sharedMemPerBlock / 1024.f),
               gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);
    }

    CUDALOG("image is %dx%d (%d samples desired)\n", w, h, ns);

#if CUDA_USE_STREAMS
    cudaStream_t d_stream[MAX_GPU];
#endif

    num_gpus = min(num_gpus, MAX_GPU);

    size_t* d_raycount[MAX_GPU];
    float3* d_output_cuda[MAX_GPU];
    float* h_output_cuda[MAX_GPU];

#if CUDA_USE_MULTIGPU
    for (int i = num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaStreamCreateWithFlags(&d_stream[i], cudaStreamNonBlocking));
#endif

        checkCudaErrors(cudaMalloc((void**)&d_output_cuda[i], w * h * sizeof(float3)));
        checkCudaErrors(cudaMemset((void*)d_output_cuda[i], 0, w * h * sizeof(float3)));

        checkCudaErrors(cudaMalloc((void**)&d_raycount[i], sizeof(size_t)));

        checkCudaErrors(cudaMallocHost((void**)&h_output_cuda[i], w * h * sizeof(float3)));
    }

#if CUDA_USE_MULTIGPU
    for (int i = num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

        int threads_per_row = sqrt(gpu_properties[i].maxThreadsPerBlock);
        int block_depth = 1;

#if CUDA_USE_MULTIGPU
        int gpu_split = num_gpus;
#else
        int gpu_split = 1;
#endif

        dim3 dimBlock(threads_per_row, threads_per_row, 1);
        dim3 dimGrid(w / dimBlock.x + 1, h / dimBlock.y + 1, block_depth);
        
        CUDALOG("raytrace<<<(%d,%d,%d), (%d,%d,%d)>>> on gpu %d\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z, i);
#if CUDA_USE_STREAMS
        raytrace<<<dimGrid, dimBlock, 0, d_stream[i]>>>(w, h, ns / block_depth / gpu_split, d_output_cuda[i], d_raycount[i]);
#else
        raytrace<<<dimGrid, dimBlock>>>(w, h, ns / block_depth / gpu_split, d_output_cuda[i], d_raycount[i]);
#endif
    }

#if CUDA_USE_MULTIGPU
    for (int i = num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaMemcpyAsync(h_output_cuda[i], d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost, d_stream[i]));

        checkCudaErrors(cudaStreamSynchronize(d_stream[i]));
#else
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(h_output_cuda[i], d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost));
#endif
      
#if CUDA_USE_MULTIGPU
        for (int j = 0; j < w * h * 3; ++j)
        {
            out_buffer[j] += h_output_cuda[i][j];
        }
#else
        memcpy(out_buffer, h_output_cuda[i], w * h * 3 * sizeof(float));
#endif

        size_t tmp;
        checkCudaErrors(cudaMemcpy(&tmp, d_raycount[i], sizeof(size_t), cudaMemcpyDeviceToHost));
        raycount += tmp;

        CUDALOG("cuda compute (%d/%d) completed!\n", i, num_gpus - 1);

        checkCudaErrors(cudaFree(d_raycount[i]));
        checkCudaErrors(cudaFree(d_output_cuda[i]));
        checkCudaErrors(cudaFreeHost(h_output_cuda[i]));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaStreamDestroy(d_stream[i]));
#endif
    }
}
