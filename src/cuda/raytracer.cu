/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include <cstdio>
#include <cfloat>
#include <cstdint>

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

void ensure(cudaError_t val, const char *const file, int const line)
{
    if (val != cudaSuccess)
    {
        CUDALOG("[CUDA error] at %s:%d code=%d (%s)\n", file, line, static_cast<unsigned int>(val), cudaGetErrorName(val));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaAssert(val) ensure((val), __FILE__, __LINE__)

//
// ----------------------------------------------------------------------------
//

__device__ inline float schlick(float costheta, float ior)
{
    float r0 = (1.f - ior) / (1.f + ior);
    r0 *= r0;
    return r0 + (1.f - r0) * powf(max(.0f, (1.f - costheta)), 5);
}

__device__ inline float fastrand(curandState* curand_ctx)
{
    return curand_uniform(curand_ctx);
}

__device__ inline float3 random_on_unit_sphere(curandState* curand_ctx)
{
    float z = fastrand(curand_ctx) * 2.f - 1.f;
    float a = fastrand(curand_ctx) * 2.f * PI;
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
};

__device__ inline float3 ray_point_at(const DRay& ray, float t)
{
    return ray.origin + t * ray.direction;
}

struct DMaterial;
struct DIntersection
{
    enum Type { eSPHERE, eBOX, eTRIANGLE };

    Type type;
    int index;
    float t;
    float3 point;
    float3 normal;
    float2 uv;
    DMaterial* material;
};

struct DMaterial
{
    enum Type { eLAMBERTIAN, eMETAL, eDIELECTRIC, eISOTROPIC, eEMISSIVE, MAX_TYPES };

    Type type;
    float3 albedo;
    float roughness;
    float ior;
};

__host__ __device__ inline DMaterial* create_material(DMaterial::Type type, float3 albedo, float roughness, float ior)
{
    DMaterial* material = new DMaterial;

    material->type = type;
    material->albedo = albedo;
    material->roughness = roughness;
    material->ior = ior;

    return material;
}

__device__ inline bool material_scatter(DMaterial& material, const DRay& ray, const DIntersection& hit, float3& attenuation, float3& emission, DRay& scattered, curandState* curand_ctx)
{
    switch (material.type)
    {

    case DMaterial::eLAMBERTIAN:
    {
        float3 target = hit.point + hit.normal + random_on_unit_sphere(curand_ctx);
        scattered.origin = hit.point;
        scattered.direction = normalize(target - hit.point);
        attenuation = material.albedo;
        emission = make_float3(.0f, .0f, .0f);

        return true;
    }

    case DMaterial::eMETAL:
    {
        float3 reflected = reflect(ray.direction, hit.normal);
        scattered.origin = hit.point;
        scattered.direction = reflected + material.roughness * random_on_unit_sphere(curand_ctx);

        attenuation = material.albedo;
        emission = make_float3(.0f, .0f, .0f);

        return (dot(scattered.direction, hit.normal) > .0f);
    }

    case DMaterial::eDIELECTRIC:
    {
        float3 outward_normal;
        attenuation = { 1.f, 1.f, 1.f };
        emission = make_float3(.0f, .0f, .0f);

        float ni_nt;
        float cosine;
        if (dot(ray.direction, hit.normal) > .0f)
        {
            outward_normal = -1.f * hit.normal;
            ni_nt = material.ior;
            cosine = dot(ray.direction, hit.normal);
            cosine = sqrtf(1.f - material.ior * material.ior * (1.f - cosine - cosine));
        }
        else
        {
            outward_normal = hit.normal;
            ni_nt = 1.f / material.ior;
            cosine = -dot(ray.direction, hit.normal);
        }

        float3 refracted;
        bool is_refracted = refract(ray.direction, outward_normal, ni_nt, refracted);
        float reflect_chance = (is_refracted) ? schlick(cosine, material.ior) : 1.0f;

        scattered.origin = hit.point;
        scattered.direction = (fastrand(curand_ctx) < reflect_chance) ? reflect(ray.direction, hit.normal) : refracted;

        return true;
    }

    case DMaterial::eEMISSIVE:
    {
        emission = material.albedo;

        return false;
    }

    default:
        return false;

    };
}

struct DSphere
{
    float3 center;
    float radius;
    DMaterial material;
};

__host__ __device__ inline DSphere* sphere_create(float3 c, float r, DMaterial mat)
{
    DSphere* sphere = new DSphere;
    sphere->center = c;
    sphere->radius = r;
    sphere->material = mat;

    return sphere;
}

__device__ inline float2 sphere_uv(DSphere& sphere, const float3& point)
{
    float phi = atan2f(point.z, point.x);
    float theta = asinf(point.y);

    return make_float2(1.0f - (phi + PI) / (2.0f * PI), (theta + PI / 2.0f) / PI);
}

__device__ inline void sphere_hit_data(DSphere& sphere, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = (hit.point - sphere.center) / sphere.radius;
    hit.uv = sphere_uv(sphere, hit.normal);
    hit.material = &sphere.material;
}

struct DBox
{
    float3 min_limit;
    float3 max_limit;
    DMaterial material;
};
    
__host__ __device__ inline DBox* box_create(float3 min, float3 max, DMaterial mat)
{
    DBox* box = new DBox;
    box->min_limit = min;
    box->max_limit = max;
    box->material = mat;

    return box;
}

__device__ inline float3 box_normal(DBox& box, const float3& point)
{
    static constexpr float eps = 1e-6f;

    if (fabs(box.min_limit.x - point.x) < eps) return make_float3(-1.f,  .0f,  .0f);
    if (fabs(box.max_limit.x - point.x) < eps) return make_float3( 1.f,  .0f,  .0f);
    if (fabs(box.min_limit.y - point.y) < eps) return make_float3( .0f, -1.f,  .0f);
    if (fabs(box.max_limit.y - point.y) < eps) return make_float3( .0f,  1.f,  .0f);
    if (fabs(box.min_limit.z - point.z) < eps) return make_float3( .0f,  .0f, -1.f);
    return make_float3(.0f, .0f, 1.f);
}

__device__ inline float2 box_uv(DBox& box, const float3& point)
{
    static constexpr float eps = 1e-6f;

    if ((fabsf(box.min_limit.x - point.x) < eps) || (fabsf(box.max_limit.x - point.x) < eps))
    {
        return make_float2((point.y - box.min_limit.y) / (box.max_limit.y - box.min_limit.y), (point.z - box.min_limit.z) / (box.max_limit.z - box.min_limit.z));
    }
    if ((fabsf(box.min_limit.y - point.y) < eps) || (fabsf(box.max_limit.y - point.y) < eps))
    {
        return make_float2((point.x - box.min_limit.x) / (box.max_limit.x - box.min_limit.x), (point.z - box.min_limit.z) / (box.max_limit.z - box.min_limit.z));
    }
    return make_float2((point.x - box.min_limit.x) / (box.max_limit.x - box.min_limit.x), (point.y - box.min_limit.y) / (box.max_limit.y - box.min_limit.y));
}

__device__ inline void box_hit_data(DBox& box, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = box_normal(box, hit.point);
    hit.uv = box_uv(box, hit.point);
    hit.material = &box.material;
}

struct DTriangle
{
    float3 vertices[3];
    float3 normal[3];
    float2 uv[3];
    float3 v0v1;
    float3 v0v2;
    DMaterial material;
};

__host__ __device__ inline DTriangle* triangle_create(float3 v1, float3 v2, float3 v3, DMaterial mat)
{
    DTriangle* triangle = new DTriangle;
    triangle->vertices[0] = v1;
    triangle->vertices[1] = v2;
    triangle->vertices[2] = v3;
    triangle->material = mat;

    triangle->v0v1 = v2 - v1;
    triangle->v0v2 = v3 - v1;
    triangle->normal[0] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->normal[1] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->normal[2] = normalize(cross(triangle->v0v1, triangle->v0v2));
    triangle->uv[0] = make_float2( .0f,  .0f);
    triangle->uv[1] = make_float2(1.0f,  .0f);
    triangle->uv[2] = make_float2( .0f, 1.0f);

    return triangle;
}

__device__ inline void triangle_hit_data(DTriangle& triangle, const DRay& ray, DIntersection& hit)
{
    hit.point = ray_point_at(ray, hit.t);
    hit.normal = triangle.normal[0]; //(1.f - hit.uv.x - hit.uv.y) * triangle.normal[0] + hit.uv.x * triangle.normal[1] + hit.uv.y * triangle.normal[2];
    hit.uv = (1.f - hit.uv.x - hit.uv.y) * triangle.uv[0] + hit.uv.x * triangle.uv[1] + hit.uv.y * triangle.uv[2];
    hit.material = &triangle.material;
}

struct DCamera
{
    float3 pos;
    float3 horizontal;
    float3 vertical;
    float3 origin;
};

__host__ __device__ inline DCamera* camera_create(const float3& eye, const float3& center, const float3& up, float fov, float ratio)
{
    DCamera* camera = new DCamera;

    float theta = radians(fov);
    float height_2 = tanf(theta / 2.f);
    float width_2 = height_2 * ratio;

    float3 w = normalize(eye - center);
    float3 u = normalize(cross(up, w));
    float3 v = cross(w, u);

    camera->pos = eye;
    camera->horizontal = 2.f * width_2 * u;
    camera->vertical = 2.f * height_2 * v;
    camera->origin = eye - width_2 * u - height_2 * v - w;

    return camera;
}

__device__ inline DRay camera_get_ray(const DCamera& camera, float s, float t)
{
    DRay ray;

    ray.origin = camera.pos;
    ray.direction = normalize(camera.origin + s * camera.horizontal + t * camera.vertical - camera.pos);

    return ray;
}

__device__ bool intersect_spheres(const DRay& ray, const DSphere* spheres, int sphere_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < sphere_count; ++i)
    {
        const DSphere& sphere = spheres[i];

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
                if (t0 > 0.01f && t0 < hit_data.t)
                {
                    hit_data.t = t0;
                    hit_data.type = DIntersection::eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }

                float t1 = -b + discriminant;
                if (t1 > 0.01f && t1 < hit_data.t)
                {
                    hit_data.t = t1;
                    hit_data.type = DIntersection::eSPHERE;
                    hit_data.index = i;
                    hit_something = true;
                }
            }
        }
    }

    return hit_something;
}

__device__ bool intersect_boxes(const DRay& ray, const DBox* boxes, int box_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < box_count; ++i)
    {
        const DBox& box = boxes[i];

        float tmin = 0.01f;
        float tmax = FLT_MAX;

        bool boxhit = false;

        #pragma unroll
        for (int side = 0; side < 3; ++side)
        {
            // TODO: think something better
            float direction = (side == 0)? ray.direction.x : (side == 1)? ray.direction.y : ray.direction.z;
            float origin = (side == 0)? ray.origin.x : (side == 1)? ray.origin.y : ray.origin.z;
            float minbound = (side == 0)? box.min_limit.x : (side == 1)? box.min_limit.y : box.min_limit.z;
            float maxbound = (side == 0)? box.max_limit.x : (side == 1)? box.max_limit.y : box.max_limit.z;

            if (fabs(direction) < 1e-6f)
            {
                if (origin < minbound || origin > maxbound)
                {
                    boxhit = false;
                    break;
                }
            }
            else
            {
                float ood = 1.f / direction;
                float t1 = (minbound - origin) * ood;
                float t2 = (maxbound - origin) * ood;

                if (t1 > t2)
                {
                    swap(t1, t2);
                }

                tmin = max(tmin, t1);
                tmax = min(tmax, t2);

                if (tmin > tmax || tmin > hit_data.t)
                {
                    boxhit = false;
                    break;
                }
                boxhit = true;
            }
        }

        if (boxhit)
        {
            hit_data.t = tmin;
            hit_data.type = DIntersection::eBOX;
            hit_data.index = i;
            hit_something = true;
        }
    }

    return hit_something;
}

__device__ bool intersect_triangles(const DRay& ray, const DTriangle* triangles, int triangle_count, DIntersection& hit_data)
{
    bool hit_something = false;

    for (int i = 0; i < triangle_count; ++i)
    {
        const DTriangle& triangle = triangles[i];
        {
            float3 pvec = cross(ray.direction, triangle.v0v2);
            float det = dot(triangle.v0v1, pvec);
    
            // if the determinant is negative the triangle is backfacing
            // if the determinant is close to 0, the ray misses the triangle
            if (det < 1e-6f)
            {
                continue;
            }
    
            float inv_det = 1.f / det;
    
            float3 tvec = ray.origin - triangle.vertices[0];
            float u = dot(tvec, pvec) * inv_det;
            if (u < .0f || u > 1.f)
            {
                continue;
            }
    
            float3 qvec = cross(tvec, triangle.v0v1);
            float v = dot(ray.direction, qvec) * inv_det;
            if (v < .0f || u + v > 1.f)
            {
                continue;
            }
    
            float t = dot(triangle.v0v2, qvec) * inv_det;
            if (t < hit_data.t)
            {
                hit_data.t = t;
                hit_data.uv = make_float2(u, v);
                hit_data.type = DIntersection::eTRIANGLE;
                hit_data.index = i;
                hit_something = true;
            }
        }
    }

    return hit_something;
}

__device__ const int MAX_DEPTH = 5;
__device__ const float3 WHITE = {1.f, 1.f, 1.f};
__device__ const float3 BLACK = {0.f, 0.f, 0.f};


__device__ float3 get_color_for(DRay ray, DSphere* spheres, int sphere_count, DBox* boxes, int box_count, DTriangle* triangles, int tri_count, int* raycount, curandState* curand_ctx)
{
    float3 total_color = WHITE;
    DRay current_ray = ray;

    for (int i = 0; i < MAX_DEPTH; ++i)
    {
        //
        // check for hits
        //
        bool hitspheres = false;
        bool hitboxes = false;
        bool hittris = false;
        DIntersection hit_data;
        hit_data.t = FLT_MAX;

        hitspheres = intersect_spheres(current_ray, spheres, sphere_count, hit_data);
        hitboxes = intersect_boxes(current_ray, boxes, box_count, hit_data);
        hittris = intersect_triangles(current_ray, triangles, tri_count, hit_data);

        ++(*raycount);

        //
        // return color or continue
        //
        if (hitspheres || hitboxes || hittris)
        {
            //
            // debug - show normals
            //
            //return .5f * (1.f + normalize(hit_data.normal));

            if (hit_data.type == DIntersection::eSPHERE)
            {
                sphere_hit_data(spheres[hit_data.index], current_ray, hit_data);
            }
            else if (hit_data.type == DIntersection::eBOX)
            {
                box_hit_data(boxes[hit_data.index], current_ray, hit_data);
            }
            else
            {
                triangle_hit_data(triangles[hit_data.index], current_ray, hit_data);
            }

            DRay scattered;
            float3 attenuation;
            float3 emission;
            if (hit_data.material && material_scatter(*hit_data.material, current_ray, hit_data, attenuation, emission, scattered, curand_ctx))
            {
                total_color *= attenuation;
                current_ray = scattered;
            }
            else
            {
                total_color *= emission;
                return total_color;
            }
        }
        else
        {
            return BLACK;
        }
    }

    return BLACK;
}

__global__ void raytrace(int width, int height, int samples, float3* pixels, int* raycount, DSphere* spheres, int spherecount, DBox* boxes, int boxcount, DTriangle* triangles, int tricount, DCamera* camera)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    curandState curand_ctx;
    curand_init(clock64(), i, 0, &curand_ctx);
    curandState* local_curand_ctx = &curand_ctx;
  
    //
    // main loop
    //
    int raycount_inst = 0;
    float3 color{ .0f, .0f, .0f };
    for (int sample = 0; sample < samples; ++sample)
    {
        float s = ((i + fastrand(local_curand_ctx)) / static_cast<float>(width));
        float t = ((j + fastrand(local_curand_ctx)) / static_cast<float>(height));

        DRay ray = camera_get_ray(*camera, s, t);
        color += get_color_for(ray, spheres, spherecount, boxes, boxcount, triangles, tricount, &raycount_inst, local_curand_ctx);
    }

    atomicAdd(raycount, raycount_inst);
    
    //
    // debug output if needed
    //
    //color.x = s;
    //color.y = t;
    //color.z = .0f;
    
    if (i < width && j < height)
    {
        float3& pixel = *(&pixels[j * width + i]);
        pixel = color;
    }

    // just to be sure we're running
    if (i == 0 && j == 0 && blockIdx.z == 0) { CUDALOG("[CUDA] running kernel...\n"); }
}

//
// IFace for raytracer.cpp
// 
#include "../scenes.hpp"

constexpr int MAX_GPU = 32;

DScene scene;

struct DCudaData
{
    bool initialized = false;
    int num_gpus = 1;
    int block_threads[MAX_GPU];
    int block_depth[MAX_GPU];
    dim3 dim_block[MAX_GPU];
    dim3 dim_grid[MAX_GPU];

    // scene definition
    DCamera* d_camera[MAX_GPU];

    DSphere* d_spheres[MAX_GPU];
    int num_spheres;

    DBox* d_boxes[MAX_GPU];
    int num_boxes;

    DTriangle* d_triangles[MAX_GPU];
    int num_triangles;

    // output buffer
    float3* d_output_cuda[MAX_GPU];
    float* h_output_cuda[MAX_GPU];

    // raycount stat
    int* d_raycount[MAX_GPU];

#if CUDA_USE_STREAMS
    cudaStream_t d_stream[MAX_GPU];
#endif
};
DCudaData data;

extern "C" void cuda_setup(const char* path, int w, int h)
{
    scene = load_scene(path, float(w) / float(h));
    
    //
    // ---- CUDA init ----
    //
    int num_gpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    data.num_gpus = min(num_gpus, MAX_GPU);

    cudaDeviceProp gpu_properties[MAX_GPU];
    for (int i = 0; i < num_gpus; i++)
    {
        checkCudaErrors(cudaGetDeviceProperties(&gpu_properties[i], i));

        CUDALOG("Device %d (%s):\n"
            "\t%d threads\n"
            "\tblocksize: %dx%dx%d\n"
            "\tshmem per block: %lu Kb\n"
            "\tgridsize: %dx%dx%d\n\n",
            i,
            gpu_properties[i].name,
            gpu_properties[i].maxThreadsPerBlock,
            gpu_properties[i].maxThreadsDim[0], gpu_properties[i].maxThreadsDim[1], gpu_properties[i].maxThreadsDim[2],
            static_cast<unsigned long>(gpu_properties[i].sharedMemPerBlock / 1024.f),
            gpu_properties[i].maxGridSize[0], gpu_properties[i].maxGridSize[1], gpu_properties[i].maxGridSize[2]);

        data.block_threads[i] = sqrt(gpu_properties[i].maxThreadsPerBlock) / 2;
        data.block_depth[i] = 1;
        
        data.dim_block[i].x = data.block_threads[i];
        data.dim_block[i].y = data.block_threads[i];
        data.dim_block[i].z = 1;
        
        data.dim_grid[i].x = w / data.dim_block[i].x + 1;
        data.dim_grid[i].y = h / data.dim_block[i].y + 1;
        data.dim_grid[i].z = data.block_depth[i];
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaStreamCreateWithFlags(&data.d_stream[i], cudaStreamNonBlocking));
#endif

        checkCudaErrors(cudaMalloc((void**)&data.d_output_cuda[i], w * h * sizeof(float3)));
        checkCudaErrors(cudaMemset((void*)data.d_output_cuda[i], 0, w * h * sizeof(float3)));
        checkCudaErrors(cudaMallocHost((void**)&data.h_output_cuda[i], w * h * sizeof(float3)));

        checkCudaErrors(cudaMalloc((void**)&data.d_raycount[i], sizeof(int)));

        //
        // ---- scene ----
        //
        checkCudaErrors(cudaMalloc((void**)&data.d_camera[i], sizeof(DCamera)));
        checkCudaErrors(cudaMemcpy(data.d_camera[i], &scene.cam, sizeof(DCamera), cudaMemcpyHostToDevice));

        data.num_spheres = scene.num_spheres;
        if (scene.num_spheres > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_spheres[i], sizeof(DSphere) * scene.num_spheres));
            for (int s = 0; s < scene.num_spheres; ++s)
            {
                checkCudaErrors(cudaMemcpy(&data.d_spheres[i][s], scene.h_spheres[s], sizeof(DSphere), cudaMemcpyHostToDevice));
            }
        }

        data.num_boxes = scene.num_boxes;
        if (scene.num_boxes > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_boxes[i], sizeof(DBox) * scene.num_boxes));
            for (int b = 0; b < scene.num_boxes; ++b)
            {
                checkCudaErrors(cudaMemcpy(&data.d_boxes[i][b], scene.h_boxes[b], sizeof(DBox), cudaMemcpyHostToDevice));
            }
        }

        data.num_triangles = scene.num_triangles;
        if (scene.num_triangles > 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data.d_triangles[i], sizeof(DTriangle) * scene.num_triangles));
            for (int t = 0; t < scene.num_triangles; ++t)
            {
                checkCudaErrors(cudaMemcpy(&data.d_triangles[i][t], scene.h_triangles[t], sizeof(DTriangle), cudaMemcpyHostToDevice));
            }
        }
    }

    data.initialized = true;
}

extern "C" void cuda_trace(int w, int h, int ns, float* out_buffer, int& out_raycount)
{
    // ensure output buffer is properly zeroed
    memset(out_buffer, 0, w * h * sizeof(float3));

    CUDALOG("image is %dx%d (%d samples desired)\n", w, h, ns);

    if (!data.initialized)
    {
        CUDALOG("CUDA scene data not initialized, aborting.\n");
        return;
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

        checkCudaErrors(cudaMemset((void*)data.d_raycount[i], 0, sizeof(int)));

        CUDALOG("raytrace<<<(%d,%d,%d), (%d,%d,%d)>>> on gpu %d\n", data.dim_block[i].x,
                                                                    data.dim_block[i].y,
                                                                    data.dim_block[i].z,
                                                                    data.dim_grid[i].x,
                                                                    data.dim_grid[i].y,
                                                                    data.dim_grid[i].z, i);

#if CUDA_USE_STREAMS
        raytrace<<<data.dim_grid[i], data.dim_block[i], 0, data.d_stream[i]>>>(w,
                                                                               h,
                                                                               ns / data.block_depth[i],
                                                                               data.d_output_cuda[i],
                                                                               data.d_raycount[i],
                                                                               data.d_spheres[i],   data.num_spheres,
                                                                               data.d_boxes[i],     data.num_boxes,
                                                                               data.d_triangles[i], data.num_triangles,
                                                                               data.d_camera[i]);

#else
        raytrace<<<data.dim_grid[i], data.dim_block[i] >>>(w,
                                                           h,
                                                           ns / data.block_depth[i],
                                                           data.d_output_cuda[i],
                                                           data.d_raycount[i],
                                                           data.d_spheres[i],   data.num_spheres,
                                                           data.d_boxes[i],     data.num_boxes,
                                                           data.d_triangles[i], data.num_triangles,
                                                           data.d_camera[i]);
#endif
        checkCudaAssert(cudaGetLastError());
    }

#if CUDA_USE_MULTIGPU
    for (int i = data.num_gpus - 1; i >= 0; --i)
    {
#else
    {
        int i = data.num_gpus - 1;
#endif
        checkCudaErrors(cudaSetDevice(i));

#if CUDA_USE_STREAMS
        checkCudaErrors(cudaMemcpyAsync(data.h_output_cuda[i], data.d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost, data.d_stream[i]));

        checkCudaErrors(cudaStreamSynchronize(data.d_stream[i]));
#else
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(data.h_output_cuda[i], data.d_output_cuda[i], w * h * sizeof(float3), cudaMemcpyDeviceToHost));
#endif
      
#if CUDA_USE_MULTIGPU
        int gpu_split = data.num_gpus;

        for (int j = 0; j < w * h * 3; ++j)
        {
            out_buffer[j] += data.h_output_cuda[i][j] / gpu_split;
        }
#else
        memcpy(out_buffer, data.h_output_cuda[i], w * h * 3 * sizeof(float));
#endif

        size_t tmp;
        checkCudaErrors(cudaMemcpy(&tmp, data.d_raycount[i], sizeof(int), cudaMemcpyDeviceToHost));
        out_raycount += tmp;

        CUDALOG("cuda compute (%d/%d) completed!\n", i, data.num_gpus - 1);
    }
}

extern "C" void cuda_cleanup()
{
    if (data.initialized)
    {
#if CUDA_USE_MULTIGPU
        for (int i = data.num_gpus - 1; i >= 0; --i)
        {
#else
        {
            int i = data.num_gpus - 1;
#endif
            checkCudaErrors(cudaSetDevice(i));

            checkCudaErrors(cudaFree(data.d_raycount[i]));
            checkCudaErrors(cudaFree(data.d_output_cuda[i]));
            checkCudaErrors(cudaFreeHost(data.h_output_cuda[i]));

            if (data.num_spheres > 0)
            {
                checkCudaErrors(cudaFree(data.d_spheres[i]));
            }

            if (data.num_boxes > 0)
            {
                checkCudaErrors(cudaFree(data.d_boxes[i]));
            }

            if (data.num_triangles > 0)
            {
                checkCudaErrors(cudaFree(data.d_triangles[i]));
            }

#if CUDA_USE_STREAMS
            checkCudaErrors(cudaStreamDestroy(data.d_stream[i]));
#endif
        }
    }
}
