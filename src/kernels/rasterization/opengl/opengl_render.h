/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "scene.h"

class OpenGLRender
{
public:
	~OpenGLRender();
	OpenGLRender(const OpenGLRender&) = delete;
	OpenGLRender& operator=(const OpenGLRender) = delete;

	static OpenGLRender& GetInstance()
	{
		static OpenGLRender instance;
		return instance;
	}

	void Initialize(Handle in_window, int in_width, int in_height, const Scene& in_scene);
	void Shutdown() {}
	void UpdateScene() {}
	void RenderScene();
	void OnPaint() {}

	const char* GetName() const { return "OpenGL"; }

	int GetRayCount() const { return 0; }

	void ResetRayCount() {}

private:
	OpenGLRender();
	bool IsInitialized() const { return init_; }
	
	Handle win_handle_{};
	int win_width_{};
	int win_height_{};
	int frame_counter_{};
	int raycount_{};

	bool init_{ false };

	const Camera* camera_{};

	struct Details;
	Details* details_{};
};
