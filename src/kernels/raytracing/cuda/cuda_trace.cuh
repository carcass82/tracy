/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
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

	uint32_t* raycount_{};

	Camera* camera_{};
	Mesh* meshes_{};
	Material* materials_{};
	Texture* textures_{};

	uint32_t meshcount_{};
	uint32_t materialcount_{};
	uint32_t texturecount_{};

	constexpr const Camera& GetCamera() const               { return *camera_; }
	constexpr const Mesh& GetMesh(uint32_t i) const         { return meshes_[i]; }
	constexpr const Material& GetMaterial(uint32_t i) const { return materials_[i]; }
	constexpr const Texture& GetTexture(uint32_t i) const   { return textures_[i]; }

	constexpr uint32_t GetMeshCount() const                 { return meshcount_; }
	constexpr uint32_t GetMaterialCount() const             { return materialcount_; }
	constexpr uint32_t GetTextureCount() const              { return texturecount_; }
};

struct HostData
{
	uint32_t width;
	uint32_t height;
	dim3 block{};
	dim3 grid{};

	cudaGraphicsResource* output_resource{};

	uint32_t frame_counter_{};
	uint32_t raycount{};
};

class CUDATraceKernel
{
public:

	bool Setup(RenderData* in_RenderData);

	bool SetupScene(const Scene& in_Scene);

	void Shutdown();

	void Trace();

	void UpdateCamera(const Camera& in_Camera);

	uint32_t GetRayCount() const { return host_data_.raycount; }

	void ResetRayCount()         { host_data_.raycount = 0; }

private:

	HostData host_data_{};

	KernelData kernel_data_{};

};
