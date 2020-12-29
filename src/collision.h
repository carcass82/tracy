/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "ray.h"
#include "mesh.h"

class Material;

namespace collision
{
struct alignas(64) HitData
{
	uint32_t object_index;
	uint32_t triangle_index;
	float t;
	vec2 uv;
	vec3 point;
	vec3 normal;
	vec3 tangent;
	uint32_t material;
};

struct TriangleHitData
{
	constexpr TriangleHitData(float t = 0) : RayT{ t } {}

	float RayT{};
	vec2 TriangleUV{};
};

struct MeshHitData : public TriangleHitData
{
	constexpr MeshHitData(float t = 0) : TriangleHitData(t) {}

	uint32_t TriangleIndex{};
};


// test ray against triangle and store result in TriangleHitData
constexpr inline bool RayTriangle(const vec3& ray_origin, const vec3& ray_direction, const vec3& in_v0, const vec3& in_v1, const vec3& in_v2, TriangleHitData& inout_hit)
{
	const vec3 v0v1 = in_v1 - in_v0;
	const vec3 v0v2 = in_v2 - in_v0;

	vec3 pvec = cross(ray_direction, v0v2);
	vec3 tvec = ray_origin - in_v0;

	float det = dot(v0v1, pvec);

	// if det is 0 the ray lies in plane of triangle and if < 0 triangle is backfacing
	if (det < EPS)
	{
		return false;
	}

	// calculate U parameter and test bounds
	float u = dot(tvec, pvec);
	if (u < EPS || u > det)
	{
		return false;
	}

	// calculate V parameter and test bounds
	vec3 qvec = cross(tvec, v0v1);
	float v = dot(ray_direction, qvec);
	if (v < EPS || u + v > det)
	{
		return false;
	}

	// ray intersects triangle, test if closer than input t
	float t = dot(v0v2, qvec) * rcp(det);
	if (t < EPS || t > inout_hit.RayT)
	{
		return false;
	}

	inout_hit.RayT = t;
	inout_hit.TriangleUV = vec2{ u, v } * rcp(det);
	return true;
}

constexpr inline bool RayTriangle(const vec3& in_ray_origin, const vec3& in_ray_direction, const vec3 in_vertices[3], TriangleHitData& inout_hit)
{
	return RayTriangle(in_ray_origin, in_ray_direction, in_vertices[0], in_vertices[1], in_vertices[2], inout_hit);
}

constexpr inline bool RayTriangle(const Ray& in_ray, const vec3 in_vertices[3], TriangleHitData& inout_hit)
{
	return RayTriangle(in_ray.GetOrigin(), in_ray.GetDirection(), in_vertices, inout_hit);
}

// test ray against triangle mesh and store result in MeshHitData
constexpr inline bool RayMesh(const Ray& in_ray, const Mesh& in_mesh, MeshHitData& inout_hit)
{
	bool result = false;

	const vec3 ray_origin{ in_ray.GetOrigin() };
	const vec3 ray_direction{ in_ray.GetDirection() };

	for (uint32_t i = 0; i < in_mesh.GetTriCount(); ++i)
	{
		const vec3 vertices[]
		{
			in_mesh.GetVertex(in_mesh.GetIndex(i * 3 + 0)).pos,
			in_mesh.GetVertex(in_mesh.GetIndex(i * 3 + 1)).pos,
			in_mesh.GetVertex(in_mesh.GetIndex(i * 3 + 2)).pos
		};

		TriangleHitData hit(inout_hit.RayT);
		if (RayTriangle(ray_origin, ray_direction, vertices, hit))
		{
			inout_hit.RayT = hit.RayT;
			inout_hit.TriangleUV = hit.TriangleUV;
			inout_hit.TriangleIndex = i * 3;
			
			result = true;
		}
	}

	return result;
}

// Fast, Branchless Ray/Bounding Box Intersections
// https://tavianator.com/fast-branchless-raybounding-box-intersections/
CUDA_ANY inline bool RayAABB(const vec3& in_ray_origin, const vec3& in_ray_inv_dir, const vec3& in_aabb_min, const vec3& in_aabb_max, float in_tmax = FLT_MAX)
{
	const vec3 minbound = (in_aabb_min - in_ray_origin) * in_ray_inv_dir;
	const vec3 maxbound = (in_aabb_max - in_ray_origin) * in_ray_inv_dir;

	const vec3 tmin1 = pmin(minbound, maxbound);
	const vec3 tmax1 = pmax(minbound, maxbound);

	const float tmin = max(tmin1.x, max(tmin1.y, tmin1.z));
	const float tmax = min(tmax1.x, min(tmax1.y, tmax1.z));

	return (tmax >= max(EPS, tmin) && tmin < in_tmax);
}

CUDA_ANY inline bool RayAABB(const Ray& in_ray, const BBox& in_aabb, float in_tmax = FLT_MAX)
{
	return RayAABB(in_ray.GetOrigin(), in_ray.GetDirectionInverse(), in_aabb.minbound, in_aabb.maxbound, in_tmax);
}

// triangle - box test using separating axis theorem (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf)
// code adapted from http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
inline bool TriangleAABB(const vec3& in_v0, const vec3& in_v1, const vec3& in_v2, const BBox& in_aabb)
{
	const vec3 v0 = in_v0;
	const vec3 v1 = in_v1;
	const vec3 v2 = in_v2;

	const vec3 e0 = v1 - v0;
	const vec3 e1 = v2 - v1;
	const vec3 e2 = v0 - v2;

	const vec3 fe0(abs(e0.x), abs(e0.y), abs(e0.z));
	const vec3 fe1(abs(e1.x), abs(e1.y), abs(e1.z));
	const vec3 fe2(abs(e2.x), abs(e2.y), abs(e2.z));

	const vec3 aabb_hsize = in_aabb.GetSize() / 2.f;

	auto AxisTester = [](float a, float b, float fa, float fb, float v0_0, float v0_1, float v1_0, float v1_1, float hsize_0, float hsize_1)
	{
		float p0 = a * v0_0 + b * v0_1;
		float p1 = a * v1_0 + b * v1_1;

		float rad = fa * hsize_0 + fb * hsize_1;
		return (min(p0, p1) > rad || max(p0, p1) < -rad);
	};

	if (AxisTester( e0.z, -e0.y, fe0.z, fe0.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
		AxisTester(-e0.z,  e0.x, fe0.z, fe0.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
		AxisTester( e0.y, -e0.x, fe0.y, fe0.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y) ||

		AxisTester( e1.z, -e1.y, fe1.z, fe1.y, v0.y, v0.z, v2.y, v2.z, aabb_hsize.y, aabb_hsize.z) ||
		AxisTester(-e1.z,  e1.x, fe1.z, fe1.x, v0.x, v0.z, v2.x, v2.z, aabb_hsize.x, aabb_hsize.z) ||
		AxisTester( e1.y, -e1.x, fe1.y, fe1.x, v0.x, v0.y, v1.x, v1.y, aabb_hsize.x, aabb_hsize.y) ||

		AxisTester( e2.z, -e2.y, fe2.z, fe2.y, v0.y, v0.z, v1.y, v1.z, aabb_hsize.y, aabb_hsize.z) ||
		AxisTester(-e2.z,  e2.x, fe2.z, fe2.x, v0.x, v0.z, v1.x, v1.z, aabb_hsize.x, aabb_hsize.z) ||
		AxisTester( e2.y, -e2.x, fe2.y, fe2.x, v1.x, v1.y, v2.x, v2.y, aabb_hsize.x, aabb_hsize.y))
	{
		return false;
	}

	vec3 trimin = pmin(v0, pmin(v1, v2));
	vec3 trimax = pmax(v0, pmax(v1, v2));
	if ((trimin.x > aabb_hsize.x || trimax.x < -aabb_hsize.x) ||
		(trimin.y > aabb_hsize.y || trimax.y < -aabb_hsize.y) ||
		(trimin.z > aabb_hsize.z || trimax.z < -aabb_hsize.z))
	{
		return false;
	}

	vec3 trinormal = cross(e0, e1);
	vec3 vmin, vmax;

	vmin.x = (trinormal.x > .0f)? -aabb_hsize.x - v0.x :  aabb_hsize.x - v0.x;
	vmax.x = (trinormal.x > .0f)?  aabb_hsize.x - v0.x : -aabb_hsize.x - v0.x;
	
	vmin.y = (trinormal.y > .0f)? -aabb_hsize.y - v0.y :  aabb_hsize.y - v0.y;
	vmax.y = (trinormal.y > .0f)?  aabb_hsize.y - v0.y : -aabb_hsize.y - v0.y;

	vmin.z = (trinormal.z > .0f)? -aabb_hsize.z - v0.z :  aabb_hsize.z - v0.z;
	vmax.z = (trinormal.z > .0f)?  aabb_hsize.z - v0.z : -aabb_hsize.z - v0.z;

	return !(dot(trinormal, vmin) > .0f || dot(trinormal, vmax) < .0f);
}

#if USE_INTRINSICS

inline __m128 _mm_hmax_ps(__m128 A)
{
	A = _mm_max_ps(A, _mm_shuffle_ps(A, A, _MM_SHUFFLE(0, 0, 2, 3)));
	A = _mm_max_ps(A, _mm_shuffle_ps(A, A, _MM_SHUFFLE(0, 0, 0, 1)));
	return A;
}

inline __m128 _mm_hmin_ps(__m128 A)
{
	A = _mm_min_ps(A, _mm_shuffle_ps(A, A, _MM_SHUFFLE(0, 0, 2, 3)));
	A = _mm_min_ps(A, _mm_shuffle_ps(A, A, _MM_SHUFFLE(0, 0, 0, 1)));
	return A;
}

inline __m128 _mm_hadd_ps(__m128 a)
{
	__m128 sum_x_y = _mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 1)));
	__m128 z = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 2));
	__m128 sum_xy_z = _mm_add_ss(sum_x_y, z);

	return sum_xy_z;
}

inline __m128 _mm_cross_ps(__m128 a, __m128 b)
{
	__m128 a_first = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 2, 1));
	__m128 b_first = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 0, 2));

	__m128 a_second = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 2));
	__m128 b_second = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 2, 1));

	return _mm_sub_ps(_mm_mul_ps(a_first, b_first), _mm_mul_ps(a_second, b_second));
}

inline __m128 _mm_dot_ps(__m128 a, __m128 b)
{
	return _mm_hadd_ps(_mm_mul_ps(a, b));
}

inline bool RayTriangle(__m128 in_ray_origin, __m128 in_ray_direction, const __m128 in_v[3], TriangleHitData& inout_hit)
{
	__m128 v0v1 = _mm_sub_ps(in_v[1], in_v[0]);
	__m128 v0v2 = _mm_sub_ps(in_v[2], in_v[0]);

	__m128 pvec = _mm_cross_ps(in_ray_direction, v0v2);
	__m128 tvec = _mm_sub_ps(in_ray_origin, in_v[0]);
	__m128 qvec = _mm_cross_ps(tvec, v0v1);

	__m128 d = _mm_dot_ps(v0v1, pvec);
	__m128 inv_d = _mm_rcp_ss(d);

	float det = _mm_cvtss_f32(d);
	float u = _mm_cvtss_f32(_mm_mul_ps(_mm_dot_ps(tvec, pvec), inv_d));
	float v = _mm_cvtss_f32(_mm_mul_ps(_mm_dot_ps(in_ray_direction, qvec), inv_d));
	float t = _mm_cvtss_f32(_mm_mul_ps(_mm_dot_ps(v0v2, qvec), inv_d));

	if (!(det < EPS || (u < EPS || u > 1.f) || (v < EPS || u + v > 1.f) || (t < EPS || t > inout_hit.RayT)))
	{
		inout_hit.RayT = t;
		inout_hit.TriangleUV.s = u;
		inout_hit.TriangleUV.t = v;
		return true;
	}
	return false;
}

inline bool RayAABB(__m128 RayOrigin, __m128 RayInvDir, __m128 BoxMin, __m128 BoxMax, float in_tmax = FLT_MAX)
{
	__m128 MinBound = _mm_mul_ps(RayInvDir, _mm_sub_ps(BoxMin, RayOrigin));
	__m128 MaxBound = _mm_mul_ps(RayInvDir, _mm_sub_ps(BoxMax, RayOrigin));

	__m128 tNear = _mm_min_ps(MinBound, MaxBound);
	__m128 tFar = _mm_max_ps(MinBound, MaxBound);

	float tmin = _mm_cvtss_f32(_mm_hmax_ps(tNear));
	float tmax = _mm_cvtss_f32(_mm_hmin_ps(tFar));

	return (tmax >= max(EPS, tmin) && tmin < in_tmax);
}

#endif

}
