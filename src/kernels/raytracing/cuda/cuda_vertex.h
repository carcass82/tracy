/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

struct CUDAVertex
{
	CUDAVertex() {}

	CUDAVertex(const vec3& in_pos, const vec3& in_normal, const vec2& in_uv0)
		: pos{ in_pos }
		, normal{ in_normal }
		, uv0{ in_uv0 }
	{}

	vec3 pos;
	vec3 normal;
	vec2 uv0;
};
