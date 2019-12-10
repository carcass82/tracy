/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "scene.h"

class CUDATrace
{
public:
	~CUDATrace();
	CUDATrace(const CUDATrace&) = delete;
	CUDATrace& operator=(const CUDATrace) = delete;

	static CUDATrace& GetInstance()
	{
		static CUDATrace instance;
		return instance;
	}

	void Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene);
	void Shutdown() {}
	void UpdateScene() {}
	void RenderScene() {}
	void OnPaint() {}

	const char* GetName() const          { return "CUDA"; }
	const int GetSamplesPerPixel() const { return samples_; }
	int GetRayCount() const              { return raycount_; }
	void ResetRayCount()                 { raycount_ = 0; }

private:
	CUDATrace();

	const int samples_{ 1 };
	const int bounces_{ 5 };

	Handle win_handle_{};
	int win_width_{};
	int win_height_{};
	int frame_counter_{};
	int raycount_{};

	const Scene* scene_{};
	const Camera* camera_{};

	struct CUDATraceDetails;
	CUDATraceDetails* details_{};
};