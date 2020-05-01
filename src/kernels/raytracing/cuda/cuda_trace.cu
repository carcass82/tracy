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
#include "cuda_log.h"

#include "ray.h"
#include "camera.h"
#include "material.h"
#include "cuda_material.h"
#include "cuda_mesh.h"
#include "scene.h"
#include "cuda_scene.h"

// static material map initialization
unordered_map<const Material*, Material*> CUDAMaterial::device_materials_;

// max gpu supported
constexpr int MAX_GPU = 32;

// max depth for ray bounces
constexpr int MAX_DEPTH = 5;

// max sample per kernel launch
constexpr int MAX_SAMPLES = 1;

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
            if (det > EPS)
            {
                vec3 tvec = ray_origin - v0;
                float u = dot(tvec, pvec);
                if (u < EPS || u > det)
                {
                    continue;
                }

                vec3 qvec = cross(tvec, v0v1);
                float v = dot(ray_direction, qvec);
                if (v < EPS || u + v > det)
                {
                    continue;
                }

                float invDet = rcp(det);
                float t = dot(v0v2, qvec) * invDet;
                if (t < inout_intersection.t && t > EPS)
                {
                    inout_intersection.t = t;
                    inout_intersection.uv = vec2{ u, v } * invDet;
                    inout_intersection.triangle_index = i * 3;
                    hit_triangle = true;
                }
            }
        }
    }

    return hit_triangle;
}

#if USE_KDTREE
__device__ bool ComputeIntersection(CUDAMesh* in_objects, int objectcount, accel::Node<Triangle, Vector>* in_scenetree, const Ray& ray, HitData& intersection_data)
#else
__device__ bool ComputeIntersection(CUDAMesh* in_objects, int objectcount, const Ray& ray, HitData& intersection_data)
#endif
{
    bool hit_any_mesh = false;

#if USE_KDTREE

    auto TriangleRayTester = [](const auto* in_triangles, unsigned int in_first, unsigned int in_count, const Ray& in_ray, HitData& intersection_data)
    {
        bool hit_triangle = false;

        const vec3 ray_direction = in_ray.GetDirection();
        const vec3 ray_origin = in_ray.GetOrigin();

        for (size_t idx = in_first; idx < in_count; ++idx)
        {
            const vec3 v0 = in_triangles[idx].v[0];
            const vec3 v0v1 = in_triangles[idx].v0v1;
            const vec3 v0v2 = in_triangles[idx].v0v2;

            vec3 pvec = cross(ray_direction, v0v2);

            float det = dot(v0v1, pvec);

            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            if (det > EPS)
            {
                vec3 tvec = ray_origin - v0;
                float u = dot(tvec, pvec);
                if (u < EPS || u > det)
                {
                    continue;
                }

                vec3 qvec = cross(tvec, v0v1);
                float v = dot(ray_direction, qvec);
                if (v < EPS || u + v > det)
                {
                    continue;
                }

                float inv_det = rcp(det);
                float t = dot(v0v2, qvec) * inv_det;
                if (t > EPS && t < intersection_data.t)
                {
                    intersection_data.t = t;
                    intersection_data.uv = vec2{ u, v } *inv_det;
                    intersection_data.triangle_index = in_triangles[idx].tri_idx;
                    intersection_data.object_index = in_triangles[idx].mesh_idx;
                    hit_triangle = true;
                }
            }
        }

        return hit_triangle;
    };

    hit_any_mesh = accel::IntersectsWithTree<Triangle>(in_scenetree, ray, intersection_data, TriangleRayTester);

#else

    for (int i = 0; i < objectcount; ++i)
    {
        if (IntersectsWithMesh(in_objects[i], ray, intersection_data))
        {
            intersection_data.object_index = i;
            hit_any_mesh = true;
        }
    }

#endif

    if (hit_any_mesh)
    {
        const CUDAMesh& m = in_objects[intersection_data.object_index];

        const Index i0 = m.indices_[intersection_data.triangle_index + 0];
        const Index i1 = m.indices_[intersection_data.triangle_index + 1];
        const Index i2 = m.indices_[intersection_data.triangle_index + 2];

        const Vertex v0 = m.vertices_[i0];
        const Vertex v1 = m.vertices_[i1];
        const Vertex v2 = m.vertices_[i2];

        intersection_data.point = ray.GetPoint(intersection_data.t);
        intersection_data.normal = normalize((1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.normal + intersection_data.uv.x * v1.normal + intersection_data.uv.y * v2.normal);
        intersection_data.uv = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.uv0 + intersection_data.uv.x * v1.uv0 + intersection_data.uv.y * v2.uv0;
        intersection_data.material = m.material_;
    }

    return hit_any_mesh;
}

__device__ inline vec3 TraceInternal(const Camera& in_camera,
                                     const Ray& in_ray,
                                     CUDAMesh* in_objects,
                                     int objectcount,
#if USE_KDTREE
                                     accel::Node<Triangle, Vector>* in_scenetree,
#endif
                                     const Material* in_skymaterial,
                                     int& inout_raycount,
                                     RandomCtx fastrand_ctx)
{
    vec3 current_color = { 1.f, 1.f, 1.f };
    Ray current_ray = { in_ray };

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        HitData hit_data;
        hit_data.t = FLT_MAX;

        ++inout_raycount;

#if USE_KDTREE
        if (ComputeIntersection(in_objects, objectcount, in_scenetree, current_ray, hit_data))
#else
        if (ComputeIntersection(in_objects, objectcount, current_ray, hit_data))
#endif
        {

#if DEBUG_SHOW_NORMALS
            current_color = .5f * normalize(1.f + (mat3(in_camera.GetView()) * hit_data.normal));
            return current_color;
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
            Ray dummy_ray;
            vec3 dummy_vec;
            vec3 sky_color;
            in_skymaterial->Scatter(current_ray, hit_data, dummy_vec, sky_color, dummy_ray, fastrand_ctx);

            current_color *= sky_color;
            return current_color;
        }
    }

    current_color = {};
    return current_color;
}

__global__ void Trace(Camera* in_camera,
                      CUDAMesh* in_objects,
                      int in_objectcount,
#if USE_KDTREE
                      accel::Node<Triangle, Vector>* in_scenetree,
#endif
                      Material* in_skymaterial,
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

#if USE_KDTREE
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, in_scenetree, in_skymaterial, cur_raycount, &curand_ctx);
#else
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, in_skymaterial, cur_raycount, &curand_ctx);
#endif
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

#if USE_KDTREE
__global__ void BuildCUDATree(accel::Node<Triangle, Vector>* inout_SceneTree, CUDAMesh* in_objects, int in_objectcount)
{
    // Triangle-AABB intersection
    auto TriangleAABBTester = [](const Triangle& triangle, const BBox& aabb)
    {
        // triangle - box test using separating axis theorem (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf)
        // code adapted from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt

        vec3 v0{ triangle.v[0] - aabb.GetCenter() };
        vec3 v1{ triangle.v[1] - aabb.GetCenter() };
        vec3 v2{ triangle.v[2] - aabb.GetCenter() };

        vec3 e0{ v1 - v0 };
        vec3 e1{ v2 - v1 };
        vec3 e2{ v0 - v2 };

        vec3 fe0{ abs(e0.x), abs(e0.y), abs(e0.z) };
        vec3 fe1{ abs(e1.x), abs(e1.y), abs(e1.z) };
        vec3 fe2{ abs(e2.x), abs(e2.y), abs(e2.z) };

        vec3 aabb_hsize = aabb.GetSize() / 2.f;

        auto AxisTester = [](float a, float b, float fa, float fb, float v0_0, float v0_1, float v1_0, float v1_1, float hsize_0, float hsize_1)
        {
            float p0 = a * v0_0 + b * v0_1;
            float p1 = a * v1_0 + b * v1_1;

            float rad = fa * hsize_0 + fb * hsize_1;
            return (min(p0, p1) > rad || max(p0, p1) < -rad);
        };

        if (AxisTester(e0.z, -e0.y, fe0.z, fe0.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
            AxisTester(-e0.z, e0.x, fe0.z, fe0.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
            AxisTester(e0.y, -e0.x, fe0.y, fe0.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y) ||

            AxisTester(e1.z, -e1.y, fe1.z, fe1.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
            AxisTester(-e1.z, e1.x, fe1.z, fe1.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
            AxisTester(e1.y, -e1.x, fe1.y, fe1.x, v0.x, v0.y, v1.x, v1.y, aabb_hsize.x, aabb_hsize.y) ||

            AxisTester(e2.z, -e2.y, fe2.z, fe2.y, v0.y, v0.z, v1.y, v1.z, aabb_hsize.y, aabb_hsize.z) ||
            AxisTester(-e2.z, e2.x, fe2.z, fe2.x, v0.x, v0.z, v1.x, v1.z, aabb_hsize.x, aabb_hsize.z) ||
            AxisTester(e2.y, -e2.x, fe2.y, fe2.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y))
        {
            return false;
        }

        vec3 trimin = pmin(v0, pmin(v1, v2));
        vec3 trimax = pmax(v0, pmax(v1, v2));
        if ((trimin.x > aabb_hsize.x || trimax.x < -aabb_hsize.x) ||
            (trimin.y > aabb_hsize.y || trimax.y < -aabb_hsize.y) ||
            (trimin.z > aabb_hsize.z || trimax.z < -aabb_hsize.z))
        {
            return false;
        }

        {
            vec3 trinormal = cross(e0, e1);

            vec3 vmin, vmax;

            if (trinormal.x > .0f) { vmin.x = -aabb_hsize.x - v0.x; vmax.x = aabb_hsize.x - v0.x; }
            else { vmin.x = aabb_hsize.x - v0.x; vmax.x = -aabb_hsize.x - v0.x; }

            if (trinormal.y > .0f) { vmin.y = -aabb_hsize.y - v0.y; vmax.y = aabb_hsize.y - v0.y; }
            else { vmin.y = aabb_hsize.y - v0.y; vmax.y = -aabb_hsize.y - v0.y; }

            if (trinormal.z > .0f) { vmin.z = -aabb_hsize.z - v0.z; vmax.z = aabb_hsize.z - v0.z; }
            else { vmin.z = aabb_hsize.z - v0.z; vmax.z = -aabb_hsize.z - v0.z; }

            if (dot(trinormal, vmin) > .0f || dot(trinormal, vmax) < .0f)
            {
                return false;
            }
        }

        return true;
    };

    printf("BuildCUDATree: building triangle scene...\n");
    long long int start = clock64();
    {
        BBox scene_bbox{ FLT_MAX, -FLT_MAX };
        for (uint16_t i = 0; i < in_objectcount; ++i)
        {
            const CUDAMesh& mesh = in_objects[i];
            for (uint16_t t = 0; t < mesh.indexcount_; t += 3)
            {
                uint16_t tri = t / 3;

                const vec3& v0 = mesh.vertices_[mesh.indices_[t + 0]].pos;
                const vec3& v1 = mesh.vertices_[mesh.indices_[t + 1]].pos;
                const vec3& v2 = mesh.vertices_[mesh.indices_[t + 2]].pos;

                scene_bbox.minbound = pmin(mesh.aabb_.minbound, scene_bbox.minbound);
                scene_bbox.maxbound = pmax(mesh.aabb_.maxbound, scene_bbox.maxbound);

                inout_SceneTree->GetElements().push_back(Triangle{ v0, v1, v2, i, tri });
            }
        }

        inout_SceneTree->SetAABB(scene_bbox);
    }
    long long int end = clock64();

    // NOTE: assuming clock is 1000MHz to convert results to ms

    printf("Scene built with %d triangles in %.2fms, now building Tree...\n", inout_SceneTree->GetSize(), (end - start) / 1.e+6f);

    start = clock64();
    accel::BuildTree<Triangle, Vector>(inout_SceneTree, TriangleAABBTester);
    end = clock64();

    printf("Tree Built in %.2fms\n", (end - start) / 1.e+6f);
}
#endif

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

#if USE_KDTREE

    CUDAAssert(cudaMalloc(&out_scene->d_scenetree, sizeof(accel::Node<Triangle, Vector>)));
    BuildCUDATree<<<1, 1>>>(out_scene->d_scenetree, out_scene->d_objects_, out_scene->objectcount_);
    CUDAAssert(cudaGetLastError());

#endif

    out_scene->d_sky_ = CUDAMaterial::Convert(in_scene.GetSkyMaterial());
    
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
#if USE_KDTREE
                           scene->d_scenetree,
#endif
                           scene->d_sky_,
                           scene->d_output_,
                           scene->width,
                           scene->height,
                           scene->d_rand_state,
                           scene->d_raycount,
                           framecount);
    
    CUDAAssert(cudaGetLastError());
}
