/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "scene.h"

class DXRTrace
{
public:
	~DXRTrace();
	DXRTrace(const DXRTrace&) = delete;
	DXRTrace& operator=(const DXRTrace) = delete;

	static DXRTrace& GetInstance()
	{
		static DXRTrace instance;
		return instance;
	}

	void Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene);
	void Shutdown();
	void UpdateScene() {}
	void RenderScene();
	void OnPaint() {}

	const char* GetName() const          { return "DX12 + DXR"; }
	const int GetSamplesPerPixel() const { return samples_; }
	int GetRayCount() const              { return raycount_; }
	void ResetRayCount()                 { raycount_ = 0; }
	
private:
	DXRTrace();
	
	struct DXRTraceDetails;
	DXRTraceDetails* details_;

	const int samples_{ 1 };
	int raycount_{};
};
