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

#if defined(WIN32)
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

	constexpr Tri(uint32_t mesh_idx, uint32_t triangle_idx, const vec3& v0, const vec3& v1, const vec3& v2)
		: packed_tri_info((mesh_idx << 24) | triangle_idx), vertices{ v0, v1, v2 }
	{}

	constexpr uint32_t GetMeshId() const { return packed_tri_info >> 24; }
	constexpr uint32_t GetTriangleId() const { return packed_tri_info & 0xffffff; }

	uint32_t packed_tri_info{ 0 };
	vec3 vertices[3]{};
};
#endif

class CPUDetails
{
public:
	bool Initialize(WindowHandle ctx, uint32_t w, uint32_t h, uint32_t size);

	void Shutdown();

	bool ProcessScene(const Scene& scene);

	bool ComputeIntersection(const Scene& scene, const Ray& ray, collision::HitData& data) const;

	void UpdateOutput(uint32_t index, const vec3& color);

	void UpdateBitmap();

	vec3 Tonemap(const vec3& color);

	void Render(WindowHandle ctx, uint32_t w, uint32_t h);

	constexpr uint32_t GetTileCount() const { return tile_count_; }

private:

	RenderData render_data_{};
	uint32_t tile_count_{};
	uint64_t frame_counter_{};

#if USE_KDTREE
	static bool TriangleRayTester(const Tri* in_triangles, uint32_t in_first, uint32_t in_count, const Ray& in_ray, collision::HitData& intersection_data);

	accel::FlatTree<Tri> scene_tree_{};
#endif
};
