#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#if defined(__CUDACC__)
 #define CUDA_CALL __host__ __device__
#else
 #define CUDA_CALL
#endif

#include "common.h"
#include "log.h"

#include "ray.h"
#include "camera.h"
#include "material.h"
#include "cuda_mesh.h"
#include "scene.h"
#include "cuda_scene.h"

constexpr int MAX_GPU = 32;
constexpr int MAX_DEPTH = 5;

struct Color
{
	static_assert(sizeof(uint32_t) == 4 * sizeof(uint8_t), "u32 != 4 * u8 :/");

	__device__ constexpr Color()                 : rgba(0)       {}
	__device__ constexpr Color(uint32_t in_rgba) : rgba(in_rgba) {}

	union
	{
		struct
		{
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		};
		
		uint32_t rgba;
	};
};

__device__ constexpr inline uint32_t ToInt(vec3 color)
{
	color *= 255.99f;

	Color c;
    c.r = static_cast<uint8_t>(clamp(color.r, 0.0f, 255.0f));
    c.g = static_cast<uint8_t>(clamp(color.g, 0.0f, 255.0f));
    c.b = static_cast<uint8_t>(clamp(color.b, 0.0f, 255.0f));
	c.a = 255;

    return c.rgba;
}

__device__ constexpr inline vec3 ToFloat(uint32_t color)
{
	Color c;
	c.rgba = color;
	
	return vec3{ c.r / 255.f, c.g / 255.f, c.b / 255.f };
}

__device__ inline float fastrand(curandState* curand_ctx)
{
    return curand_uniform(curand_ctx);
}

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
        const Vertex v0 = m.vertices_[m.indices_[intersection_data.triangle_index + 0]];
        const Vertex v1 = m.vertices_[m.indices_[intersection_data.triangle_index + 1]];
        const Vertex v2 = m.vertices_[m.indices_[intersection_data.triangle_index + 2]];

        intersection_data.point = ray.GetPoint(intersection_data.t);
        intersection_data.normal = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.normal + intersection_data.uv.x * v1.normal + intersection_data.uv.y * v2.normal;
        intersection_data.uv = (1.f - intersection_data.uv.x - intersection_data.uv.y) * v0.uv0 + intersection_data.uv.x * v1.uv0 + intersection_data.uv.y * v2.uv0;
        intersection_data.material = &m.material_;
    }

    return hit_any_mesh;
}

__device__ inline vec3 TraceInternal(const Camera& in_camera, const Ray& in_ray, CUDAMesh* in_objects, int objectcount, int& inout_raycount)
{
    vec3 current_color = { 1.f, 1.f, 1.f };
    Ray current_ray = { in_ray };

    HitData hit_data;
    hit_data.t = FLT_MAX;

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        ++inout_raycount;

        if (ComputeIntersection(in_objects, objectcount, current_ray, hit_data))
        {

#if DEBUG_SHOW_NORMALS
            return .5f * normalize(1.f + (mat3(in_camera.GetView()) * hit_data.normal));
#else
            Ray scattered;
            vec3 attenuation;
            vec3 emission;
            if (hit_data.material->Scatter(current_ray, hit_data, attenuation, emission, scattered))
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
                      uint32_t* output,
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
    int old_raycount = raycount[j * width + i];

    int cur_raycount = 0;

    vec3 cur_color{};
    for (int sample = 0; sample < 1; ++sample)
    {
        float s = ((i + fastrand(&curand_ctx)) / f_width);
        float t = ((j + fastrand(&curand_ctx)) / f_height);
    
        Ray r = in_camera->GetRayFrom(s, t);
        cur_color += TraceInternal(*in_camera, r, in_objects, in_objectcount, cur_raycount);
    }
    
    rand_state[j * width + i] = curand_ctx;
    raycount[j * width + i] = old_raycount + cur_raycount;

	const float blend_factor = framecount / static_cast<float>(framecount + 1);
    vec3 old_color = ToFloat(output[j * width + i]);
    output[j * width + i] = ToInt(lerp(cur_color, old_color, blend_factor));
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

    CUDAAssert(cudaSetDevice(0));

    //
    // copy data to the device
    //

    CUDAAssert(cudaMalloc(&out_scene->objects_, in_scene.GetObjectCount() * sizeof(CUDAMesh)));
    for (int i = 0; i < in_scene.GetObjectCount(); ++i)
    {
        CUDAMesh cmesh(in_scene.GetObject(i));
        CUDAAssert(cudaMemcpy(&out_scene->objects_[i], &cmesh, sizeof(CUDAMesh), cudaMemcpyHostToDevice));
    }
    out_scene->objectcount_ = in_scene.GetObjectCount();

    CUDAAssert(cudaMalloc(&out_scene->d_camera_, sizeof(Camera)));
    CUDAAssert(cudaMemcpy(out_scene->d_camera_, &in_scene.GetCamera(), sizeof(Camera), cudaMemcpyHostToDevice));

    CUDAAssert(cudaMalloc(&out_scene->d_rand_state, out_scene->width * out_scene->height * sizeof(curandState)));

    dim3 block(16, 16, 1);
    dim3 grid(out_scene->width / block.x + 1, out_scene->height / block.y + 1, 1);
    InitRandom<<<grid, block>>> (out_scene->d_rand_state, out_scene->width, out_scene->height);

    out_scene->h_raycount = new int[out_scene->width * out_scene->height];
    CUDAAssert(cudaMalloc(&out_scene->d_raycount, out_scene->width * out_scene->height * sizeof(int)));
    CUDAAssert(cudaMemset(out_scene->d_raycount, 0, out_scene->width * out_scene->height * sizeof(int)));
}


extern "C" void cuda_trace(CUDAScene* scene, unsigned int* output, int framecount)
{
    CUDAAssert(cudaSetDevice(0));

    dim3 block(16, 16, 1);
    dim3 grid(scene->width / block.x + 1, scene->height / block.y + 1, 1);

    Trace<<<grid, block>>>(scene->d_camera_, scene->objects_, scene->objectcount_, output, scene->width, scene->height, scene->d_rand_state, scene->d_raycount, framecount);
    
    CUDAAssert(cudaGetLastError());
}
