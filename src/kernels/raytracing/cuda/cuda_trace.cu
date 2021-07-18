/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#include "cuda_trace.cuh"
#include "cuda_details.h"
#include "cuda_log.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <curand.h>

#include "collision.h"
#include "camera.h"
#include "scene.h"

constexpr uint32_t kMaxBounces{ TRACY_MAX_BOUNCES };

__device__ bool Intersects(const Ray& ray, const KernelData& data, HitData& intersection)
{
    bool hit_any_mesh{ false };

    const vec3 ray_origin{ ray.GetOrigin() };
    const vec3 ray_invdir{ ray.GetDirectionInverse() };

    for (uint32_t i = 0; i < data.GetMeshCount(); ++i)
    {
        auto& object = data.GetMesh(i);
        auto& aabb = object.GetAABB();

        if (collision::RayAABB(ray_origin, ray_invdir, aabb.minbound, aabb.maxbound, intersection.t))
        {
            collision::MeshHitData mesh_hit(intersection.t);
            if (collision::RayMesh(ray, object, mesh_hit))
            {
                intersection.t = mesh_hit.RayT;
                intersection.uv = mesh_hit.TriangleUV;
                intersection.triangle_index = mesh_hit.TriangleIndex;
                intersection.object_index = i;
                hit_any_mesh = true;
            }
        }
    }

    if (hit_any_mesh)
    {
        const Mesh& mesh = data.GetMesh(intersection.object_index);
    
        const Index i0 = mesh.GetIndex(intersection.triangle_index + 0);
        const Index i1 = mesh.GetIndex(intersection.triangle_index + 1);
        const Index i2 = mesh.GetIndex(intersection.triangle_index + 2);
    
        const Vertex v0 = mesh.GetVertex(i0);
        const Vertex v1 = mesh.GetVertex(i1);
        const Vertex v2 = mesh.GetVertex(i2);
    
        const vec2 uv = intersection.uv;
    
        intersection.point = ray.GetPoint(intersection.t);
        intersection.normal = normalize((1.f - uv.x - uv.y) * v0.normal + uv.x * v1.normal + uv.y * v2.normal);
        intersection.tangent = (1.f - uv.x - uv.y) * v0.tangent + uv.x * v1.tangent + uv.y * v2.tangent;
        intersection.uv = (1.f - uv.x - uv.y) * v0.uv0 + uv.x * v1.uv0 + uv.y * v2.uv0;
        intersection.material = mesh.GetMaterial();
    }
    
    return hit_any_mesh;
}


__device__ vec3 Trace(Ray&& ray, const KernelData& data, RandomCtx random_ctx)
{
    Ray current_ray{ std::move(ray) };
    vec3 throughput{ 1.f, 1.f, 1.f };
    vec3 pixel;

    uint32_t raycount{};

    for (uint32_t t = 0; t < kMaxBounces; ++t)
    {
        ++raycount;

        HitData intersection_data;
        intersection_data.t = FLT_MAX;

        vec3 attenuation;
        vec3 emission;

        if (Intersects(current_ray, data, intersection_data))
        {

#if DEBUG_SHOW_BASECOLOR
            return data.GetMaterial(intersection_data.material).GetBaseColor(data, intersection_data);
#elif DEBUG_SHOW_NORMALS
            return .5f * normalize((1.f + mat3(data.GetCamera().GetView()) * data.GetMaterial(intersection_data.material).GetNormal(data, intersection_data)));
#elif DEBUG_SHOW_METALNESS
            return vec3(data.GetMaterial(intersection_data.material).GetMetalness(data, intersection_data));
#elif DEBUG_SHOW_ROUGHNESS
            return vec3(data.GetMaterial(intersection_data.material).GetRoughness(data, intersection_data));
#elif DEBUG_SHOW_EMISSIVE
            return data.GetMaterial(intersection_data.material).GetEmissive(data, intersection_data);
#endif

            data.GetMaterial(intersection_data.material).Scatter(data, current_ray, intersection_data, attenuation, emission, current_ray, random_ctx);
            {
                pixel += emission * throughput;
                throughput *= attenuation;
            }
        }
        else
        {
            const vec3 v{ current_ray.GetDirection() };
            intersection_data.uv = vec2(atan2f(v.z, v.x) / (2 * PI), asinf(v.y) / PI) + 0.5f;
            emission = data.GetMaterial(Scene::SKY_MATERIAL_ID).GetEmissive(data, intersection_data);

            pixel += emission * throughput;
            break;
        }

#if USE_RUSSIAN_ROULETTE
        float p = EPS + max(throughput.r, max(throughput.g, throughput.b));
        if (fastrand(random_ctx) > p)
        {
            break;
        }

        throughput *= rcp(p);
#endif
    }

    atomicAdd(data.raycount_, raycount);
    return pixel;
}

//
// Kernels
//
__global__ void TraceKernel(cudaSurfaceObject_t surface, KernelData data, uint32_t w, uint32_t h, uint32_t frame_count)
{
    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t j = (blockIdx.y * blockDim.y) + threadIdx.y;
    const uint32_t idx = j * w + i;

    if LIKELY(i < w && j < h)
    {
        RandomCtxData curand = data.randctx_[idx];
        RandomCtx curand_ctx = &curand;

        const float s = ((i + fastrand(curand_ctx)) / static_cast<float>(w));
        const float t = ((j + fastrand(curand_ctx)) / static_cast<float>(h));
        const vec4 color{ Trace(data.GetCamera().GetRayFrom(s, t), data, curand_ctx), 1.f };
        
        data.randctx_[idx] = curand;

#if ACCUMULATE_SAMPLES

        const float blend_factor{ frame_count / (frame_count + 1.f) };
        
        const vec4 prev_color{ surf2Dread<float4>(surface, i * sizeof(float4), j) };

        surf2Dwrite<float4>(lerp(color, prev_color, blend_factor), surface, i * sizeof(float4), j);

#else

        surf2Dwrite<float4>(color, surface, i * sizeof(float4), j);

#endif
    }
}

__global__ void InitRandom(KernelData data, uint32_t w, uint32_t h)
{
    const uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t j = (blockIdx.y * blockDim.y) + threadIdx.y;
    const uint32_t idx = j * w + i;

    if LIKELY(i < w && j < h)
    {
        curand_init(0x12345, idx, 0, &data.randctx_[idx]);
    }
}
//
// End Kernels
//

void CUDATraceKernel::Trace()
{
    CUDAAssert(cudaGraphicsMapResources(1, &host_data_.output_resource, 0));
    
    cudaArray_t output_array{};
    CUDAAssert(cudaGraphicsSubResourceGetMappedArray(&output_array, host_data_.output_resource, 0, 0));
    
    cudaResourceDesc description{};
    description.resType = cudaResourceTypeArray;
    description.res.array.array = output_array;
    
    cudaSurfaceObject_t surface_object;
    CUDAAssert(cudaCreateSurfaceObject(&surface_object, &description));
    
    TraceKernel<<<host_data_.grid, host_data_.block>>>(surface_object, kernel_data_, host_data_.width, host_data_.height, host_data_.frame_counter_++);

    CUDAAssert(cudaDestroySurfaceObject(surface_object));
    
    CUDAAssert(cudaGraphicsUnmapResources(1, &host_data_.output_resource, 0));
    
    // TODO: find something better to keep track of raycount
    uint32_t raycount;
    CUDAAssert(cudaMemcpy(&raycount, kernel_data_.raycount_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    host_data_.raycount += raycount;

    CUDAAssert(cudaStreamSynchronize(0));

    CUDAAssert(cudaMemsetAsync(kernel_data_.raycount_, 0, sizeof(uint32_t)));
}

void CUDATraceKernel::UpdateCamera(const Camera& in_Camera)
{
    host_data_.frame_counter_ = 0;
    CUDAAssert(cudaMemcpy(kernel_data_.camera_, &in_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
}

bool CUDATraceKernel::Setup(RenderData* in_RenderData)
{
    if (in_RenderData)
    {
        const uint32_t w{ in_RenderData->width };
        const uint32_t h{ in_RenderData->height };
        const dim3 block{ 8, 8 };
        const dim3 grid{ (w + block.x - 1) / block.x, (h + block.y - 1) / block.y };

        host_data_.width = w;
        host_data_.height = h;
        host_data_.block = block;
        host_data_.grid = grid;

        CUDAAssert(cudaGraphicsGLRegisterImage(&host_data_.output_resource,
                                               in_RenderData->output_texture,
                                               GL_TEXTURE_2D,
                                               cudaGraphicsRegisterFlagsSurfaceLoadStore));

        // rand seed generation on device
        CUDAAssert(cudaMalloc(&kernel_data_.randctx_, host_data_.width * host_data_.height * sizeof(curandState)));

        InitRandom<<<host_data_.grid, host_data_.block>>>(kernel_data_, host_data_.width, host_data_.height);

        CUDAAssert(cudaMalloc(&kernel_data_.raycount_, sizeof(uint32_t)));
        CUDAAssert(cudaMemset(kernel_data_.raycount_, 0, sizeof(uint32_t)));
        
        CUDAAssert(cudaDeviceSynchronize());

        return true;
    }

    return false;
}

bool CUDATraceKernel::SetupScene(const Scene& in_Scene)
{
    CUDAAssert(cudaMalloc(&kernel_data_.camera_, sizeof(Camera)));
    CUDAAssert(cudaMemcpy(kernel_data_.camera_, &in_Scene.GetCamera(), sizeof(Camera), cudaMemcpyHostToDevice));

    kernel_data_.meshcount_ = in_Scene.GetObjectCount();
    CUDAAssert(cudaMalloc(&kernel_data_.meshes_, kernel_data_.meshcount_ * sizeof(Mesh)));
    for (uint32_t i = 0; i < kernel_data_.meshcount_; ++i)
    {
        auto& host_mesh = in_Scene.GetObject(i);

        Vertex* vertices{};
        uint32_t vertexcount{ host_mesh.GetVertexCount() };
        CUDAAssert(cudaMalloc(&vertices, vertexcount * sizeof(Vertex)));
        CUDAAssert(cudaMemcpy(vertices, host_mesh.GetVertices(), vertexcount * sizeof(Vertex), cudaMemcpyHostToDevice));

        Index* indices{};
        uint32_t indexcount{ host_mesh.GetIndexCount() };
        CUDAAssert(cudaMalloc(&indices, indexcount * sizeof(Index)));
        CUDAAssert(cudaMemcpy(indices, host_mesh.GetIndices(), indexcount * sizeof(Index), cudaMemcpyHostToDevice));

        Mesh* mesh = new Mesh(vertices, vertexcount, indices, indexcount, host_mesh.GetAABB(), host_mesh.GetMaterial());
        CUDAAssert(cudaMemcpy(&kernel_data_.meshes_[i], mesh, sizeof(Mesh), cudaMemcpyHostToDevice));
    }

    kernel_data_.materialcount_ = static_cast<uint32_t>(in_Scene.GetMaterials().size());
    CUDAAssert(cudaMalloc(&kernel_data_.materials_, kernel_data_.materialcount_ * sizeof(Material)));
    CUDAAssert(cudaMemcpy(kernel_data_.materials_, in_Scene.GetMaterials().data(), kernel_data_.materialcount_ * sizeof(Material), cudaMemcpyHostToDevice));

    kernel_data_.texturecount_ = static_cast<uint32_t>(in_Scene.GetTextures().size());
    CUDAAssert(cudaMalloc(&kernel_data_.textures_, kernel_data_.texturecount_ * sizeof(Texture)));
    for (uint32_t i = 0; i < kernel_data_.texturecount_; ++i)
    {
        auto&& host_texture = in_Scene.GetTexture(i);
        uint32_t host_texture_size = host_texture.GetWidth() * host_texture.GetHeight();

        vec4* pixels{};
        CUDAAssert(cudaMalloc(&pixels, host_texture_size * sizeof(vec4)));
        CUDAAssert(cudaMemcpy(pixels, host_texture.GetPixels(), host_texture_size * sizeof(vec4), cudaMemcpyHostToDevice));

        Texture* texture = new Texture(host_texture.GetWidth(), host_texture.GetHeight(), pixels);
        CUDAAssert(cudaMemcpy(&kernel_data_.textures_[i], texture, sizeof(Texture), cudaMemcpyHostToDevice));
    }

    CUDAAssert(cudaDeviceSynchronize());

    return true;
}

void CUDATraceKernel::Shutdown()
{
    CUDAAssert(cudaGraphicsUnregisterResource(host_data_.output_resource));

    CUDAAssert(cudaFree(kernel_data_.raycount_));

    CUDAAssert(cudaFree(kernel_data_.randctx_));

    CUDAAssert(cudaFree(kernel_data_.textures_));

    CUDAAssert(cudaFree(kernel_data_.materials_));

    CUDAAssert(cudaFree(kernel_data_.meshes_));

    CUDAAssert(cudaFree(kernel_data_.camera_));
}
