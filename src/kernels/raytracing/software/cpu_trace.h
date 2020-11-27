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
	void OnUpdate(const Scene& in_Scene, float in_DeltaTime) override;
	void OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene) override;
	void OnRender(const WindowHandle in_Window) override;
	void Shutdown() override;
	const char* GetModuleName() const override { return "CPU"; }

	int GetRayCount() const       { return raycount_; }
	void ResetRayCount()          { raycount_ = 0; }

private:
	vec3 Trace(const Ray& ray, const Scene& scene, RandomCtx random_ctx);

#if TILED_RENDERING
	void RenderTile(uint32_t tile_x, uint32_t tile_y, uint32_t tile_size, const Scene& scene, RandomCtx random_ctx);
#endif

	static constexpr uint32_t kTileSize{ 4 };
	static constexpr uint32_t kBounces{ 5 };
	
	int raycount_{};
};
