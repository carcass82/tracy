/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#if USE_GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/compatibility.hpp>
using glm::mat4;
using glm::mat3;
using glm::vec4;
using glm::vec3;
using glm::vec2;
using glm::radians;
using glm::max;
using glm::min;
using glm::clamp;
using glm::lerp;
using glm::perspective;
using glm::lookAt;
constexpr float PI = 3.1415926535897932f;
#else
#include "cclib/cclib.h"
using cc::math::mat4;
using cc::math::mat3;
using cc::math::vec4;
using cc::math::vec3;
using cc::math::vec2;
using cc::math::radians;
using cc::util::max;
using cc::util::min;
using cc::util::clamp;
using cc::math::lerp;
using cc::math::perspective;
using cc::math::inverse;
using cc::math::PI;
#endif

class Material;
struct HitData
{
	float t;
	vec2 uv;
	vec3 point;
	vec3 normal;
	const Material* material;
};

struct BBox
{
	BBox()
		: minbound{}
		, maxbound{}
	{}

	BBox(const vec3& in_minbound, const vec3& in_maxbound)
		: minbound(in_minbound)
		, maxbound(in_maxbound)
	{}

	vec3 minbound;
	vec3 maxbound;
};

constexpr inline uint32_t make_id(char a, char b, char c = '\0', char d = '\0')
{
	return a | b << 8 | c << 16 | d << 24;
}

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOSERVICE
#define NOMCX
#include <Windows.h>

using Handle = HWND;

#else

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#endif
