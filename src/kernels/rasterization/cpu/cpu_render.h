/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#include "common.h"
#include "module.h"

class CPURaster final : public TracyModule<CPURaster>
{
	friend class TracyModule<CPURaster>;

public:
	bool Startup(const WindowHandle in_Window, const Scene& in_Scene) override;
	void OnUpdate(const Scene& in_Scene, float in_DeltaTime) override;
	void OnEvent(TracyEvent in_Event, const WindowHandle in_Window, const Scene& in_Scene) override {}
	void OnRender(const WindowHandle in_Window) override;
	void Shutdown() override;
	const char* GetModuleName() const override { return "CPU"; }

	uint32_t GetRayCount() const { return 0; }
	void ResetRayCount() {}
};
