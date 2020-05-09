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
#include "scene.h"
#include "cuda_scene.h"
#include "collision.h"

// static material map initialization
unordered_map<const Material*, Material*> CUDAScene::device_materials_;

// max depth for ray bounces
constexpr int MAX_DEPTH = 5;

// max sample per kernel launch
constexpr int MAX_SAMPLES = 1;


#if USE_KDTREE
__device__ bool ComputeIntersection(Mesh* in_objects, int objectcount, CUDATree* in_scenetree, const Ray& ray, HitData& intersection_data)
#else
__device__ bool ComputeIntersection(Mesh* in_objects, int objectcount, const Ray& ray, HitData& intersection_data)
#endif
{
    bool hit_any_mesh = false;

#if USE_KDTREE

    auto TriangleRayTester = [&in_objects](const auto* in_triangles, unsigned int in_first, unsigned int in_count, const Ray& in_ray, HitData& intersection_data)
    {
        bool hit_triangle = false;

        const vec3 ray_direction = in_ray.GetDirection();
        const vec3 ray_origin = in_ray.GetOrigin();

        for (size_t idx = in_first; idx < in_count; ++idx)
        {
            const uint32_t mesh_id = in_triangles[idx].GetMeshId();
            const uint32_t triangle_id = in_triangles[idx].GetTriangleId() * 3;

            const auto& mesh = in_objects[mesh_id];

            const vec3 v0 = mesh.GetVertex(mesh.GetIndex(triangle_id + 0)).pos;
            const vec3 v1 = mesh.GetVertex(mesh.GetIndex(triangle_id + 1)).pos;
            const vec3 v2 = mesh.GetVertex(mesh.GetIndex(triangle_id + 2)).pos;

            collision::TriangleHitData tri_hit_data(intersection_data.t);
            if (collision::RayTriangle(in_ray, v0, v1, v2, tri_hit_data))
            {
                intersection_data.t = tri_hit_data.RayT;
                intersection_data.uv = tri_hit_data.TriangleUV;
                intersection_data.triangle_index = triangle_id;
                intersection_data.object_index = mesh_id;
                hit_triangle = true;
            }
        }

        return hit_triangle;
    };

    hit_any_mesh = accel::IntersectsWithTree<TriInfo>(in_scenetree->GetChild(0), ray, intersection_data, TriangleRayTester);

#else

    for (int i = 0; i < objectcount; ++i)
    {
        if (collision::RayAABB(ray, in_objects[i].GetAABB(), intersection_data.t))
        {
            collision::MeshHitData mesh_hit(intersection_data.t);
            if (collision::RayMesh(ray, in_objects[i], mesh_hit))
            {
                intersection_data.t = mesh_hit.RayT;
                intersection_data.uv = mesh_hit.TriangleUV;
                intersection_data.triangle_index = mesh_hit.TriangleIndex;
                intersection_data.object_index = i;
                hit_any_mesh = true;
            }
        }
    }

#endif

    if (hit_any_mesh)
    {
        const auto& m = in_objects[intersection_data.object_index];

        const Index i0 = m.GetIndex(intersection_data.triangle_index + 0);
        const Index i1 = m.GetIndex(intersection_data.triangle_index + 1);
        const Index i2 = m.GetIndex(intersection_data.triangle_index + 2);

        const Vertex v0 = m.GetVertex(i0);
        const Vertex v1 = m.GetVertex(i1);
        const Vertex v2 = m.GetVertex(i2);

        intersection_data.point = ray.GetPoint(intersection_data.t);
        intersection_data.normal = normalize((1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.normal + intersection_data.uv.x * v1.normal + intersection_data.uv.y * v2.normal);
        intersection_data.uv = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.uv0 + intersection_data.uv.x * v1.uv0 + intersection_data.uv.y * v2.uv0;
        intersection_data.material = m.GetMaterial();
    }

    return hit_any_mesh;
}

__device__ inline vec3 TraceInternal(const Camera& in_camera,
                                     const Ray& in_ray,
                                     Mesh* in_objects,
                                     int objectcount,
#if USE_KDTREE
                                     CUDATree* in_scenetree,
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
                      Mesh* in_objects,
                      int in_objectcount,
#if USE_KDTREE
                      CUDATree* in_scenetree,
#endif
                      Material* in_skymaterial,
                      vec4* out_tempresult,
                      int in_width,
                      int in_height,
                      curandState* inout_rand_state,
                      int* inout_raycount,
                      int in_framecount)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int idx = j * in_width + i;

    if (i >= in_width || j >= in_height)
    {
        return;
    }

    curandState curand_ctx = inout_rand_state[idx];
    
    int cur_raycount = 0;

    vec3 cur_color{};
    for (int sample = 0; sample < MAX_SAMPLES; ++sample)
    {
        float s = ((i + fastrand(&curand_ctx)) / (float)in_width);
        float t = ((j + fastrand(&curand_ctx)) / (float)in_height);
    
        Ray r = in_camera->GetRayFrom(s, t);

#if USE_KDTREE
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, in_scenetree, in_skymaterial, cur_raycount, &curand_ctx);
#else
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, in_skymaterial, cur_raycount, &curand_ctx);
#endif
    }
    cur_color /= MAX_SAMPLES;
    
    // update random seed
    inout_rand_state[idx] = curand_ctx;

    // update raycount
    atomicAdd(inout_raycount, cur_raycount);

	float blend_factor = in_framecount / float(in_framecount + 1);
    vec3 old_color = out_tempresult[idx].rgb;
    out_tempresult[idx] = vec4(lerp(cur_color, old_color, blend_factor), 1.f);
}

__global__ void InitRandom(int device, curandState* rand_state, int width, int height)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    curand_init(clock64() * device, i, j, &rand_state[j * width + i]);
}

extern "C" void cuda_setup(const Scene& in_scene, CUDAScene* out_scene)
{
    int gpucount;
    CUDAAssert(cudaGetDeviceCount(&gpucount));

#if CUDA_FORCE_SINGLE_DEVICE
    out_scene->num_gpus = 1;
#else
    out_scene->num_gpus = min(gpucount, MAX_GPU);
#endif

    TracyLog("Found %d %s, %d will be used\n\n", gpucount, ((gpucount > 1) ? "devices" : "device"), out_scene->num_gpus);

    
    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < out_scene->num_gpus; i++)
    {
        CUDAAssert(cudaGetDeviceProperties(&gpu_properties[i], i));

        CUDALog("Device %d (%s):\n"
                "\t%d threads\n"
                "\tblocksize: %dx%dx%d\n"
                "\tshmem per block: %dKb\n"
                "\tgridsize: %dx%dx%d\n",
                i,
                gpu_properties[i].name,
                gpu_properties[i].maxThreadsPerBlock,
                gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
                gpu_properties[i].sharedMemPerBlock / 1024,
                gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);

        for (int j = 0; j < out_scene->num_gpus; j++)
        {
            if (i != j)
            {
                CUDAAssert(cudaSetDevice(i));

                int result = -1;
                cudaDeviceCanAccessPeer(&result, i, j);

                CUDAAssert(cudaDeviceEnablePeerAccess(j, 0));

                TracyLog("\t* enabled p2p access to Device %d\n", i, j);
            }
        }

        TracyLog("\n\n");
    }

    CUDAAssert(cudaSetDevice(0));

    // copy camera to device
    {
        CUDAAssert(cudaMalloc(&out_scene->d_camera_, sizeof(Camera)));
        CUDAAssert(cudaMemcpy(out_scene->d_camera_, &in_scene.GetCamera(), sizeof(Camera), cudaMemcpyHostToDevice));
    }
    
    // copy all meshes and materials to device
    {
        out_scene->objectcount_ = in_scene.GetObjectCount();
        CUDAAssert(cudaMalloc(&out_scene->d_objects_, out_scene->objectcount_ * sizeof(Mesh)));
        for (int i = 0; i < out_scene->objectcount_; ++i)
        {
            const Mesh& cpumesh = in_scene.GetObject(i);

            uint32_t vertexcount = cpumesh.GetVertexCount();
            Vertex* vertices;
            CUDAAssert(cudaMalloc(&vertices, vertexcount * sizeof(Vertex)));
            CUDAAssert(cudaMemcpy(vertices, &cpumesh.GetVertices()[0], vertexcount * sizeof(Vertex), cudaMemcpyHostToDevice));

            uint32_t indexcount = cpumesh.GetIndexCount();
            Index* indices;
            CUDAAssert(cudaMalloc(&indices, indexcount * sizeof(Index)));
            CUDAAssert(cudaMemcpy(indices, &cpumesh.GetIndices()[0], indexcount * sizeof(Index), cudaMemcpyHostToDevice));

            Mesh* cudamesh = new Mesh(vertices, vertexcount, indices, indexcount, CUDAScene::ConvertMaterial(cpumesh.GetMaterial()));
            cudamesh->SetAABB(cpumesh.GetAABB());

            CUDAAssert(cudaMemcpy(&out_scene->d_objects_[i], cudamesh, sizeof(Mesh), cudaMemcpyHostToDevice));
        }

        out_scene->d_sky_ = CUDAScene::ConvertMaterial(in_scene.GetSkyMaterial());
    }

#if USE_KDTREE

    // copy the tree to the device
    {
        CUDATree helper;
        helper.nodes_num_ = out_scene->h_scenetree.nodes_num_;
        helper.elements_num_ = out_scene->h_scenetree.elements_num_;

        CUDAAssert(cudaMalloc(&helper.elements_, helper.elements_num_ * sizeof(TriInfo)));
        CUDAAssert(cudaMemcpy(helper.elements_, out_scene->h_scenetree.elements_, helper.elements_num_ * sizeof(TriInfo), cudaMemcpyHostToDevice));

        CUDAAssert(cudaMalloc(&out_scene->d_scenetree, sizeof(CUDATree)));
        for (unsigned int i = 0; i < helper.nodes_num_; ++i)
        {
            out_scene->h_scenetree.nodes_[i].root = out_scene->d_scenetree;
        }
        CUDAAssert(cudaMalloc(&helper.nodes_, helper.nodes_num_ * sizeof(CUDANode)));
        CUDAAssert(cudaMemcpy(helper.nodes_, out_scene->h_scenetree.nodes_, helper.nodes_num_ * sizeof(CUDANode), cudaMemcpyHostToDevice));

        CUDAAssert(cudaMemcpy(out_scene->d_scenetree, &helper, sizeof(CUDATree), cudaMemcpyHostToDevice));
    }

#endif

    // rand seed generation on device
    {
        CUDAAssert(cudaMalloc(&out_scene->d_rand_state, out_scene->width * out_scene->height * sizeof(curandState)));

        dim3 block(8, 8, 1);
        dim3 grid(out_scene->width / block.x + 1, out_scene->height / block.y + 1, 1);
        InitRandom<<<grid, block>>> (0x12345 + 0, out_scene->d_rand_state, out_scene->width, out_scene->height);

        CUDAAssert(cudaGetLastError());
    }

    // prepare raycount
    {
        CUDAAssert(cudaMalloc(&out_scene->d_raycount, sizeof(int)));
        CUDAAssert(cudaMemset(out_scene->d_raycount, 0, sizeof(int)));
    }
    
    // prepare output buffer
    {
    	CUDAAssert(cudaMalloc(&out_scene->d_output_, out_scene->width * out_scene->height * sizeof(vec4)));
    	CUDAAssert(cudaMemset(out_scene->d_output_, 0, out_scene->width * out_scene->height * sizeof(vec4)));
    }
}

extern "C" void cuda_trace(CUDAScene* scene, int framecount)
{
    dim3 block(8, 8, 1);
    dim3 grid(scene->width / block.x + 1, scene->height / block.y + 1, 1);

    CUDAAssert(cudaSetDevice(0));
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

extern "C" void cuda_shutdown(CUDAScene* scene)
{
    for (int i = 0; i < scene->num_gpus; i++)
    {
        for (int j = 0; j < scene->num_gpus; j++)
        {
            if (i != j)
            {
                CUDAAssert(cudaSetDevice(i));

                int result = -1;
                cudaDeviceCanAccessPeer(&result, i, j);

                CUDAAssert(cudaDeviceDisablePeerAccess(j));
            }
        }
    }
}
