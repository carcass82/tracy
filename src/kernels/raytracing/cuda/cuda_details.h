/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"

#include "GL/glew.h"
#if !defined(_WIN32)
 #include <X11/Xlib.h>
 #include <GL/glx.h>
#endif
#include <cuda_gl_interop.h>

#if USE_KDTREE
 #include "kdtree.h"
#endif

#include "cuda_trace.cuh"

class Scene;
class Camera;
class Mesh;
class Material;

struct RenderData
{
	u32 width{};
	u32 height{};

#if defined(_WIN32)
	HDC hDC;
	HGLRC hRC;
#else
    Display* dpy;
    GLXContext glCtx;
#endif

	static const char* vs_shader;
	static const char* fs_shader;

	const char* fullscreen_texture_name = "fsTex";
	GLint fullscreen_texture{};
	GLuint fullscreen_shader{};

	GLuint output_texture{};
};

#if USE_KDTREE
struct Tri
{
	constexpr Tri()
	{}

	Tri(u32 mesh_idx, u32 triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
		: packed_tri_info((mesh_idx << 24) | triangle_idx)
		, vertices{ v0, v1, v2 }
	{}

	constexpr u32 GetMeshId() const { return packed_tri_info >> 24; }
	constexpr u32 GetTriangleId() const { return packed_tri_info & 0xffffff; }

	u32 packed_tri_info{ 0 };
	vec3 vertices[3]{};
};

struct Obj
{
	constexpr Obj()
	{}

	constexpr Obj(u32 object_idx)
		: object_id{ object_idx }
	{}

	u32 object_id{};
};
#endif

class CUDADetails
{
public:
	bool Initialize(WindowHandle ctx, u32 w, u32 h);

	bool ProcessScene(const Scene& scene);

	void Update(const Scene& scene);

	void Render(WindowHandle ctx, u32 w, u32 h);

	void Shutdown();

	void CameraUpdated();

	u32 GetRayCount();

	void ResetRayCount();

private:
	void InitGLContext(WindowHandle ctx);

	RenderData render_data_{};

	CUDATraceKernel kernel_{};

	bool camera_updated_{ false };

#if USE_KDTREE
	accel::FlatTree<Obj> TLAS_tree_{};
	vector<accel::FlatTree<Tri>> BLAS_tree_{};
#endif
};
