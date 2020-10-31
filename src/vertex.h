/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"

template <bool WITH_TANGENTS_AND_BITANGENTS> struct BaseVertex
{
};

//
//
//

template<>
struct BaseVertex<true>
{
	CUDA_CALL BaseVertex(const vec3& in_pos, const vec3& in_normal, const vec2& in_uv0, const vec3& in_tangent, const vec3& in_bitangent)
		: pos{ in_pos }
		, normal{ in_normal }
		, uv0{ in_uv0 }
		, tangent{ in_tangent }
		, bitangent{ in_bitangent }
	{}

	CUDA_CALL BaseVertex(const vec3& in_pos, const vec3& in_normal = {}, const vec2& in_uv0 = {})
		: BaseVertex(in_pos, in_normal, in_uv0, vec3{}, vec3{})
	{}

	CUDA_CALL BaseVertex()
	{}

	vec3 pos{};
	vec3 normal{};
	vec2 uv0{};
	vec3 tangent{};
	vec3 bitangent{};

	static constexpr bool VertexHasTangents{ true };
};

//
//
//

template<>
struct BaseVertex<false>
{
	CUDA_CALL BaseVertex(const vec3& in_pos, const vec3& in_normal = {}, const vec2& in_uv0 = {})
		: pos{ in_pos }
		, normal{ in_normal }
		, uv0{ in_uv0 }
	{}

	CUDA_CALL BaseVertex()
	{}

	vec3 pos{};
	vec3 normal{};
	vec2 uv0{};

	static constexpr bool VertexHasTangents{ false };
};
