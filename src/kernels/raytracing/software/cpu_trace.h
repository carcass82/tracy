/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"
#include "module.h"

class Scene;
class Ray;
class Camera;

class CpuTrace final : public TracyModule
{
public:
	bool Startup(const WindowHandle in_Window, const Scene& in_Scene) override;
	void OnUpdate(const Scene& in_Scene, float in_DeltaTime) override;
	void OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene) override;
	void OnRender(const WindowHandle in_Window) override;
	void Shutdown() override;
	const char* GetModuleName() const override { return "CPU RTX"; }
	u32 GetRayCount(bool in_ShouldReset) override;

private:
	vec3 Trace(Ray&& ray, const Scene& scene, RandomCtx random_ctx);

#if TILED_RENDERING
	void RenderTile(u32 tile_x, u32 tile_y, const Scene& scene, RandomCtx random_ctx);
#endif

	static constexpr u32 kTileSize{ 4 };
	static constexpr u32 kMaxBounces{ TRACY_MAX_BOUNCES };

	u32 raycount_{};
};

inline u32 CpuTrace::GetRayCount(bool in_ShouldReset)
{
	u32 res{ raycount_ };
	
	if (in_ShouldReset)
	{
		raycount_ = 0;
	}

	return res;
}
