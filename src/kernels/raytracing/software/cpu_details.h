/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"
#include "bitmap.h"

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
	u32 width{};
	u32 height{};
	vector<vec3> output{};
	Bitmap bitmap;
};

#if USE_KDTREE
struct Tri
{
	constexpr Tri()
	{}

	Tri(u32 mesh_idx, u32 triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
		: packed_tri_info((mesh_idx << 24) | triangle_idx)
#if USE_INTRINSICS
		, vertices{ _mm_set_ps(v0.z, v0.z, v0.y, v0.x), _mm_set_ps(v1.z, v1.z, v1.y, v1.x), _mm_set_ps(v2.z, v2.z, v2.y, v2.x) }
#else
		, vertices{ v0, v1, v2 }
#endif
	{}

	constexpr u32 GetMeshId() const { return packed_tri_info >> 24; }

	constexpr u32 GetTriangleId() const { return packed_tri_info & 0xffffff; }


	u32 packed_tri_info{ 0 };

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

	constexpr Obj(u32 object_idx)
		: object_id{ object_idx }
	{}

	u32 object_id{};
};
#endif

class CPUDetails
{
public:
	bool Initialize(WindowHandle ctx, u32 w, u32 h, u32 size);

	void Shutdown();

	bool ProcessScene(const Scene& scene);

	bool ComputeIntersection(const Scene& scene, const Ray& ray, HitData& data) const;

	void UpdateOutput(u32 index, const vec3& color);

	void UpdateBitmap();

	vec3 Tonemap(const vec3& color) const;

	void Render(WindowHandle ctx, u32 w, u32 h);

	constexpr u32 GetTileCount() const { return tile_count_; }

	void ResetFrameCounter() { frame_counter_ = 0; }


private:

	RenderData render_data_{};
	
	u32 tile_count_{};
	
	u64 frame_counter_{};

#if USE_KDTREE
	
	accel::FlatTree<Obj> TLAS_tree_{};
	
	vector<accel::FlatTree<Tri>> BLAS_tree_{};

#endif
};
