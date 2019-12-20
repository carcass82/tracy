/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <cstdint>

#if USE_GLM
 #if defined(__CUDACC__)
  #define GLM_FORCE_CUDA
 #endif
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_PRECISION_MEDIUMP_FLOAT
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/fast_trigonometry.hpp>
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
#define cosf(x) glm::fastCos(x)
#define sinf(x) glm::fastSin(x)
constexpr inline vec3 pmin(const vec3& a, const vec3& b) { return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) }; }
constexpr inline vec3 pmax(const vec3& a, const vec3& b) { return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; }
constexpr float PI = glm::pi<float>();
template<typename T, size_t N> constexpr inline uint32_t array_size(const T(&)[N]) { return N; }
#define CC_CONSTEXPR
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
using cc::util::array_size;
using cc::math::lerp;
using cc::math::perspective;
using cc::math::inverse;
#define cosf(x) cc::math::fast::cosf(x)
#define sinf(x) cc::math::fast::sinf(x)
using cc::math::PI;
#define CC_CONSTEXPR constexpr
#endif

class Material;
struct HitData
{
	int object_index;
	int triangle_index;
	float t;
	
	vec2 uv;
	vec3 point;
	vec3 normal;
	const Material* material;
};

struct BBox
{
	CUDA_CALL BBox()
		: minbound{}
		, maxbound{}
	{}

	CUDA_CALL BBox(const vec3& in_minbound, const vec3& in_maxbound)
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

#if !defined(CUDA_CALL) 
 #define CUDA_CALL
#endif

#if !defined(CUDA_DEVICE_CALL)
 #define CUDA_DEVICE_CALL
#endif

#if defined(__CUDACC__)
 #include <curand_kernel.h>
 using RandomCtx = curandState*;
 #define fastrand(x) curand_uniform(x)
#else
 using RandomCtx = uint32_t&;
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOSERVICE
#define NOMCX
#include <Windows.h>
using Handle = HWND;
#define DEBUG_BREAK() __debugbreak()

#elif defined(__linux__)

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
struct handle_t
{
	int ds;
	Display* dpy;
	Window win;
};
using Handle = struct handle_t*;

#if !defined(MAX_PATH)
 #define MAX_PATH 260
#endif

#define DEBUG_BREAK() __builtin_trap()

#else

#error "only windows and linux are supported at this time!"

#endif
