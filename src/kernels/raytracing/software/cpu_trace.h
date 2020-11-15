/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "module.h"

class Scene;
class Ray;
class Camera;

class CpuTrace : public TracyModule<CpuTrace>
{
	friend class TracyModule<CpuTrace>;

public:
	bool Startup(const WindowHandle in_Window, const Scene& in_Scene) override;
	void OnUpdate(const Scene& in_Scene) override;
	void OnRender(const WindowHandle in_Window) override;
	void Shutdown() override;
	const char* GetModuleName() const override { return "CPU"; }

	int GetRayCount() const       { return raycount_; }
	void ResetRayCount()          { raycount_ = 0; }

private:
	void RenderTile(uint32_t tile_x, uint32_t tile_y, uint32_t tile_size, const Scene& scene);
	vec3 Trace(const Ray& ray, const Scene& scene, uint32_t random_ctx);

	static constexpr uint32_t kTileSize{ 8 };
	static constexpr uint32_t kBounces{ 5 };
	
	int raycount_{};
};
