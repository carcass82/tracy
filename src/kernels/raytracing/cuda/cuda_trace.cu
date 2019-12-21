/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <cfloat>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"
#include "log.h"

#include "ray.h"
#include "camera.h"
#include "material.h"
#include "cuda_mesh.h"
#include "scene.h"
#include "cuda_scene.h"

// static material map initialization
unordered_map<const Material*, Material*> CUDAMaterial::host_to_device_;

// max gpu supported
constexpr int MAX_GPU = 32;

// max depth for ray bounces
constexpr int MAX_DEPTH = 5;

// max sample per kernel launch
constexpr int MAX_SAMPLES = 1;

__device__ bool IntersectsWithBoundingBox(const BBox& box, const Ray& ray, float nearest_intersection = FLT_MAX)
{
    const vec3 inv_ray = ray.GetInvDirection();
    const vec3 minbound = (box.minbound - ray.GetOrigin()) * inv_ray;
    const vec3 maxbound = (box.maxbound - ray.GetOrigin()) * inv_ray;

    vec3 tmin1 = pmin(minbound, maxbound);
    vec3 tmax1 = pmax(minbound, maxbound);

    float tmin = max(tmin1.x, max(tmin1.y, tmin1.z));
    float tmax = min(tmax1.x, min(tmax1.y, tmax1.z));

    return (tmax >= max(1.e-8f, tmin) && tmin < nearest_intersection);
}

__device__ bool IntersectsWithMesh(const CUDAMesh& mesh, const Ray& in_ray, HitData& inout_intersection)
{
    bool hit_triangle = false;

    if (IntersectsWithBoundingBox(mesh.aabb_, in_ray, inout_intersection.t))
    {
        vec3 ray_direction = in_ray.GetDirection();
        vec3 ray_origin = in_ray.GetOrigin();

        int tris = mesh.indexcount_ / 3;
        for (int i = 0; i < tris; ++i)
        {
            const Index i0 = mesh.indices_[i * 3 + 0];
            const Index i1 = mesh.indices_[i * 3 + 1];
            const Index i2 = mesh.indices_[i * 3 + 2];

            const vec3 v0 = mesh.vertices_[i0].pos;
            const vec3 v1 = mesh.vertices_[i1].pos;
            const vec3 v2 = mesh.vertices_[i2].pos;

            const vec3 v0v1 = v1 - v0;
            const vec3 v0v2 = v2 - v0;

            vec3 pvec = cross(ray_direction, v0v2);
            float det = dot(v0v1, pvec);

            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            if (det < 1.e-8f)
            {
                continue;
            }

            float invDet = 1.f / det;

            vec3 tvec = ray_origin - v0;
            float u = dot(tvec, pvec) * invDet;
            if (u < .0f || u > 1.f)
            {
                continue;
            }

            vec3 qvec = cross(tvec, v0v1);
            float v = dot(ray_direction, qvec) * invDet;
            if (v < .0f || u + v > 1.f)
            {
                continue;
            }

            float t = dot(v0v2, qvec) * invDet;
            if (t < inout_intersection.t && t > 1.e-3f)
            {
                inout_intersection.t = dot(v0v2, qvec) * invDet;
                inout_intersection.uv = vec2{ u, v };
                inout_intersection.triangle_index = i * 3;
                hit_triangle = true;
            }
        }
    }

    return hit_triangle;
}

__device__ bool ComputeIntersection(CUDAMesh* in_objects, int objectcount, const Ray& ray, HitData& intersection_data)
{
    bool hit_any_mesh = false;
    
    for (int i = 0; i < objectcount; ++i)
    {
        if (IntersectsWithMesh(in_objects[i], ray, intersection_data))
        {
            intersection_data.object_index = i;
            hit_any_mesh = true;
        }
    }

    if (hit_any_mesh)
    {
        const CUDAMesh& m = in_objects[intersection_data.object_index];

        const Index i0 = m.indices_[intersection_data.triangle_index + 0];
        const Index i1 = m.indices_[intersection_data.triangle_index + 1];
        const Index i2 = m.indices_[intersection_data.triangle_index + 2];

        const CUDAVertex v0 = m.vertices_[i0];
        const CUDAVertex v1 = m.vertices_[i1];
        const CUDAVertex v2 = m.vertices_[i2];

        intersection_data.point = ray.GetPoint(intersection_data.t);
        intersection_data.normal = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.normal + intersection_data.uv.x * v1.normal + intersection_data.uv.y * v2.normal;
        intersection_data.uv = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.uv0 + intersection_data.uv.x * v1.uv0 + intersection_data.uv.y * v2.uv0;
        intersection_data.material = m.material_;
    }

    return hit_any_mesh;
}

__device__ inline vec3 TraceInternal(const Camera& in_camera, const Ray& in_ray, CUDAMesh* in_objects, int objectcount, int& inout_raycount, RandomCtx fastrand_ctx)
{
    vec3 current_color = { 1.f, 1.f, 1.f };
    Ray current_ray = { in_ray };

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        HitData hit_data;
        hit_data.t = FLT_MAX;

        ++inout_raycount;

        if (ComputeIntersection(in_objects, objectcount, current_ray, hit_data))
        {

#if DEBUG_SHOW_NORMALS
            return .5f * normalize(1.f + (mat3(in_camera.GetView()) * hit_data.normal));
#else
            Ray scattered;
            vec3 attenuation;
            vec3 emission;
            if (hit_data.material->Scatter(current_ray, hit_data, attenuation, emission, scattered, fastrand_ctx))
            {
                current_color *= attenuation;
                current_ray = scattered;
            }
            else
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

__global__ void Trace(Camera* in_camera,
                      CUDAMesh* in_objects,
                      int in_objectcount,
                      vec4* output_float,
                      int width,
                      int height,
                      curandState* rand_state,
                      int* raycount,
                      int framecount)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    const float f_width = static_cast<float>(width);
    const float f_height = static_cast<float>(height);

    curandState curand_ctx = rand_state[j * width + i];
    
    int cur_raycount = 0;

    vec3 cur_color{};
    for (int sample = 0; sample < MAX_SAMPLES; ++sample)
    {
        float s = ((i + fastrand(&curand_ctx)) / f_width);
        float t = ((j + fastrand(&curand_ctx)) / f_height);
    
        Ray r = in_camera->GetRayFrom(s, t);
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, cur_raycount, &curand_ctx);
    }
    cur_color /= MAX_SAMPLES;
    
    rand_state[j * width + i] = curand_ctx;    
    atomicAdd(raycount, cur_raycount);


	float blend_factor = framecount / float(framecount + 1);
    
    vec3 old_color = output_float[j * width + i].rgb;
    
    // "unroll" to use lerp version with cuda intrinsics
    output_float[j * width + i] = vec4(lerp(cur_color.r, old_color.r, blend_factor),
                                       lerp(cur_color.g, old_color.g, blend_factor),
                                       lerp(cur_color.b, old_color.b, blend_factor),
                                       1.f);
}

__global__ void InitRandom(curandState* rand_state, int width, int height)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    curand_init(clock64(), i, j, &rand_state[j * width + i]);
}

extern "C" void cuda_setup(const Scene& in_scene, CUDAScene* out_scene)
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

    CUDAAssert(cudaSetDevice(CUDA_PREFERRED_DEVICE));

    //
    // copy data to the device
    //

    CUDAAssert(cudaMalloc(&out_scene->d_objects_, in_scene.GetObjectCount() * sizeof(CUDAMesh)));
    for (int i = 0; i < in_scene.GetObjectCount(); ++i)
    {
        CUDAMesh cmesh(in_scene.GetObject(i), CUDAMaterial::Convert(in_scene.GetObject(i).GetMaterial()));
        CUDAAssert(cudaMemcpy(&out_scene->d_objects_[i], &cmesh, sizeof(CUDAMesh), cudaMemcpyHostToDevice));
    }
    out_scene->objectcount_ = in_scene.GetObjectCount();

    CUDAAssert(cudaMalloc(&out_scene->d_camera_, sizeof(Camera)));
    CUDAAssert(cudaMemcpy(out_scene->d_camera_, &in_scene.GetCamera(), sizeof(Camera), cudaMemcpyHostToDevice));

    CUDAAssert(cudaMalloc(&out_scene->d_rand_state, out_scene->width * out_scene->height * sizeof(curandState)));

    dim3 block(16, 16, 1);
    dim3 grid(out_scene->width / block.x + 1, out_scene->height / block.y + 1, 1);
    InitRandom<<<grid, block>>> (out_scene->d_rand_state, out_scene->width, out_scene->height);

    CUDAAssert(cudaGetLastError());

    CUDAAssert(cudaMalloc(&out_scene->d_raycount, sizeof(int)));
    CUDAAssert(cudaMemset(out_scene->d_raycount, 0, sizeof(int)));
}


extern "C" void cuda_trace(CUDAScene* scene, int framecount)
{
    CUDAAssert(cudaSetDevice(CUDA_PREFERRED_DEVICE));

    dim3 block(16, 16, 1);
    dim3 grid(scene->width / block.x + 1, scene->height / block.y + 1, 1);

    Trace<<<grid, block>>>(scene->d_camera_,
                           scene->d_objects_,
                           scene->objectcount_,
                           scene->d_output_,
                           scene->width,
                           scene->height,
                           scene->d_rand_state,
                           scene->d_raycount,
                           framecount);
    
    CUDAAssert(cudaGetLastError());

    cudaMemcpy(&scene->h_raycount, scene->d_raycount, sizeof(int), cudaMemcpyDeviceToHost);
}
