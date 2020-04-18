/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "scene.h"

class CpuTrace
{
public:
	~CpuTrace();
	CpuTrace(const CpuTrace&) = delete;
	CpuTrace& operator=(const CpuTrace) = delete;

	static CpuTrace& GetInstance()
	{
		static CpuTrace instance;
		return instance;
	}

	void Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene);
	void Shutdown() {}
	void UpdateScene();
	void RenderScene();
	void OnPaint();

	const char* GetName() const          { return "CPU"; }
	int GetRayCount() const              { return raycount_; }
	void ResetRayCount()                 { raycount_ = 0; }

	const Scene* GetScene() const        { return scene_; }

private:
	CpuTrace();
	vec3 Trace(const Ray& ray, uint32_t rand_ctx);

	const int bounces_{ 5 };

	Handle win_handle_{};
	int win_width_{};
	int win_height_{};
	int raycount_{};

	const Scene* scene_{};
	const Camera* camera_{};

	struct CpuTraceDetails;
	CpuTraceDetails* details_{};
};
