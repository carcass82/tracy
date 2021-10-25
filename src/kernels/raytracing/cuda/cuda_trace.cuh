/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"
#include "mesh.h"
#include "material.h"
#include "texture.h"
#include <curand_kernel.h>

class Scene;
class Camera;
struct RenderData;


struct KernelData
{
	curandState* randctx_{};

	u32* raycount_{};

	Camera* camera_{};
	Mesh* meshes_{};
	Material* materials_{};
	Texture* textures_{};

	u32 meshcount_{};
	u32 materialcount_{};
	u32 texturecount_{};

	constexpr const Camera& GetCamera() const          { return *camera_; }
	constexpr const Mesh& GetMesh(u32 i) const         { return meshes_[i]; }
	constexpr const Material& GetMaterial(u32 i) const { return materials_[i]; }
	constexpr const Texture& GetTexture(u32 i) const   { return textures_[i]; }

	constexpr u32 GetMeshCount() const                 { return meshcount_; }
	constexpr u32 GetMaterialCount() const             { return materialcount_; }
	constexpr u32 GetTextureCount() const              { return texturecount_; }
};

struct HostData
{
	u32 width;
	u32 height;
	dim3 block{};
	dim3 grid{};

	cudaGraphicsResource* output_resource{};

	u32 frame_counter_{};
	u32 raycount{};
};

class CUDATraceKernel
{
public:

	bool Setup(RenderData* in_RenderData);

	bool SetupScene(const Scene& in_Scene);

	void Shutdown();

	void Trace();

	void UpdateCamera(const Camera& in_Camera);

	u32 GetRayCount() const { return host_data_.raycount; }

	void ResetRayCount()    { host_data_.raycount = 0; }

private:

	HostData host_data_{};

	KernelData kernel_data_{};

};
