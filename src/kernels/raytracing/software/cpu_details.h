/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"

#if USE_KDTREE
 #include "kdtree.h"
#endif

using std::vector;

class Scene;
class Ray;
namespace collision { struct HitData; }

#if defined(_MSC_VER) && _MSC_VER < 1920
 // pre-vs2019 ms compiler does not support openmp "collapse" clause
 #define collapse(x) 
#endif

struct RenderData
{
	uint32_t width{};
	uint32_t height{};
	vector<vec3> output{};
	uint32_t* bitmap_bytes{};

#if defined(_WIN32)
	HBITMAP bitmap{};
#else
	XImage* bitmap{};
#endif
};

#if USE_KDTREE
struct Tri
{
	constexpr Tri()
	{}

	Tri(uint32_t mesh_idx, uint32_t triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
		: packed_tri_info((mesh_idx << 24) | triangle_idx)
#if USE_INTRINSICS
		, vertices{ _mm_set_ps(v0.z, v0.z, v0.y, v0.x), _mm_set_ps(v1.z, v1.z, v1.y, v1.x), _mm_set_ps(v2.z, v2.z, v2.y, v2.x) }
#else
		, vertices{ v0, v1, v2 }
#endif
	{}

	constexpr uint32_t GetMeshId() const { return packed_tri_info >> 24; }

	constexpr uint32_t GetTriangleId() const { return packed_tri_info & 0xffffff; }


	uint32_t packed_tri_info{ 0 };

#if USE_INTRINSICS
	__m128 vertices[3]{};
#else
	vec3 vertices[3]{};
#endif
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

class CPUDetails
{
public:
	bool Initialize(WindowHandle ctx, uint32_t w, uint32_t h, uint32_t size);

	void Shutdown();

	bool ProcessScene(const Scene& scene);

	bool ComputeIntersection(const Scene& scene, const Ray& ray, HitData& data) const;

	void UpdateOutput(uint32_t index, const vec3& color);

	void UpdateBitmap();

	vec3 Tonemap(const vec3& color);

	void Render(WindowHandle ctx, uint32_t w, uint32_t h);

	constexpr uint32_t GetTileCount() const { return tile_count_; }

	void ResetFrameCounter() { frame_counter_ = 0; }


private:

	RenderData render_data_{};
	
	uint32_t tile_count_{};
	
	uint64_t frame_counter_{};

#if USE_KDTREE
	
	accel::FlatTree<Obj> TLAS_tree_{};
	
	vector<accel::FlatTree<Tri>> BLAS_tree_{};

#endif
};
