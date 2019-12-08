/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

struct Vertex
{
	Vertex()
		: pos{}
		, normal{}
		, uv0{}
		, tangent{}
		, bitangent{}
	{}

	Vertex(const vec3& in_pos, const vec3& in_normal, const vec2& in_uv0, const vec3& in_tangent, const vec3& in_bitangent)
		: pos(in_pos)
		, normal(in_normal)
		, uv0(in_uv0)
		, tangent(in_tangent)
		, bitangent(in_bitangent)
	{}

	//
	//
	//

	vec3 pos;
	vec3 normal;
	vec2 uv0;
	vec3 tangent;
	vec3 bitangent;
};

using Index = uint32_t;
