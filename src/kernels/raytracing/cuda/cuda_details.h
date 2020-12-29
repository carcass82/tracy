/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
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
	uint32_t width{};
	uint32_t height{};

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

	Tri(uint32_t mesh_idx, uint32_t triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
		: packed_tri_info((mesh_idx << 24) | triangle_idx)
		, vertices{ v0, v1, v2 }
	{}

	constexpr uint32_t GetMeshId() const { return packed_tri_info >> 24; }
	constexpr uint32_t GetTriangleId() const { return packed_tri_info & 0xffffff; }

	uint32_t packed_tri_info{ 0 };
	vec3 vertices[3]{};
};

struct Obj
{
	constexpr Obj()
	{}

	constexpr Obj(uint32_t object_idx)
		: object_id{ object_idx }
	{}

	uint32_t object_id{};
};
#endif

class CUDADetails
{
public:
	bool Initialize(WindowHandle ctx, uint32_t w, uint32_t h);

	bool ProcessScene(const Scene& scene);

	void Update(const Scene& scene);

	void Render(WindowHandle ctx, uint32_t w, uint32_t h);

	void Shutdown();

	void CameraUpdated();

	uint32_t GetRayCount();

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
