/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <utility>
#include <string.h>

#if defined(_DEBUG)
 #define DEBUG_ASSERT(x) assert(x)
#else
 #define DEBUG_ASSERT(x)
#endif

#ifndef __has_cpp_attribute
# define __has_cpp_attribute(x) 0
#endif

// use as "if LIKELY(cond)" without parentheses surrounding LIKELY/UNLIKELY
#if __has_cpp_attribute(likely) && __has_cpp_attribute(unlikely)
 #define LIKELY(x)   (!!(x)) [[likely]]
 #define UNLIKELY(x) (!!(x)) [[unlikely]]
#elif !defined(_MSC_VER) || defined(__CUDACC__)
 #define LIKELY(x)   (__builtin_expect(!!(x), 1))
 #define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
 #define LIKELY(x)   (x)
 #define UNLIKELY(x) (x)
#endif

#if defined(_MSC_VER) && _MSC_VER < 1920
 // pre-vs2019 ms compiler does not support openmp "collapse" clause
 #define collapse(x)
#endif

// types shortcut
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8 = int8_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

#if 0 // TODO: custom container for vector/string
 #include "cclib/ccvector.h"
 template<class T>
 using vector = cc::Vector<T>;
#else
 #include <vector>
 template<class T>
 using vector = std::vector<T>;
#endif

#if defined(__CUDACC__)
 #if !defined(CUDA_ANY)
  #define CUDA_ANY __host__ __device__
 #endif
 #if !defined(CUDA_DEVICE)
  #define CUDA_DEVICE __device__
 #endif
 #if !defined(CUDA_HOST)
  #define CUDA_HOST __host__
 #endif
#else
 #define CUDA_ANY
 #define CUDA_DEVICE
 #define CUDA_HOST
#endif

#if !defined(TRACY_EXPOSURE)
 #define TRACY_EXPOSURE 1.f
#endif

#if defined(_DEBUG)
 #if defined(_MSC_VER)
  #define DEBUG_BREAK() __debugbreak()
 #elif defined(__GNUC__) || defined(__clang__)
  #define DEBUG_BREAK() __builtin_trap()
 #else
  #define DEBUG_BREAK() { /* don't know how to break with this compiler */ }
 #endif
#else
 #define DEBUG_BREAK() {}
#endif

#if defined(_MSC_VER)
 #define NOVTABLE __declspec(novtable)
#else
 #define NOVTABLE
#endif

#if USE_GLM
 #if defined(__CUDACC__)
  #include <cuda.h>
  #define GLM_FORCE_CUDA
 #else
  #define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
 #endif

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_PRECISION_MEDIUMP_FLOAT
#define GLM_FORCE_SWIZZLE

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/fast_trigonometry.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/color_space.hpp>
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
using glm::inverse;
using glm::transpose;
using glm::translate;
using glm::rotate;
using glm::scale;
using glm::lookAt;
using glm::normalize;
#define frac(x) glm::fract(x)
#define cosf(x) glm::fastCos(x)
#define sinf(x) glm::fastSin(x)
CUDA_ANY inline void sincosf(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }
CUDA_ANY inline vec3 pmin(const vec3& a, const vec3& b) { return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) }; }
CUDA_ANY inline vec3 pmax(const vec3& a, const vec3& b) { return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; }
constexpr float PI = 3.1415926535897932f;
constexpr float EPS = 1.e-8f;
template<typename T, size_t N> constexpr inline u32 array_size(const T(&)[N]) { return N; }
template<typename T> constexpr inline T rcp(const T& x) { return 1.f / x; }
#define srgb(x) convertLinearToSRGB(x)
#define linear(x) convertSRGBToLinear(x)
#else
#include "cclib/cclib.h"
using cc::math::mat4;
using cc::math::mat3;
using cc::math::vec4;
using cc::math::vec3;
using cc::math::vec2;
using cc::math::radians;
using cc::math::max;
using cc::math::min;
using cc::math::rcp;
using cc::math::clamp;
using cc::math::lerp;
using cc::math::perspective;
using cc::math::inverse;
using cc::math::transpose;
using cc::math::translate;
using cc::math::rotate;
using cc::math::scale;
using cc::math::normalize;
#define cosf(x) cc::math::cosf(x)
#define sinf(x) cc::math::sinf(x)
#define sincosf(x, s, c) cc::math::sincosf(x, s, c)
#define powf(x, y) cc::math::pow(x, y)
#define sqrtf(x) cc::math::sqrtf(x)
using cc::math::frac;
using cc::math::PI;
using cc::math::EPS;
using cc::array_size;
using cc::gfx::srgb;
using cc::gfx::linear;
#endif

#if defined(__CUDACC__)
 #include <curand_kernel.h>
 using RandomCtxData = curandState;
 using RandomCtx = curandState*;
 #define fastrand(x) curand_uniform(x)

 #include <nvfunctional>
 using nvstd::function;

#else

 using RandomCtxData = u32;
 using RandomCtx = RandomCtxData&;

 #include <functional>
 using std::function;
#endif

struct alignas(64) HitData
{
	u32 object_index;
	u32 triangle_index;
	float t;
	vec2 uv;
	vec3 point;
	vec3 normal;
	vec3 tangent;
	u32 material;
};

#include "vertex.h"
using Vertex = BaseVertex<true /* with tangent and bitangent */>;
using Index = u32;

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOSERVICE
#define NOMCX
#include <Windows.h>
struct handle_t
{
	u32 width;
	u32 height;
	HWND win;
};
using WindowHandle = struct handle_t*;

inline WindowHandle CreateWindowHandle(HWND hwnd, u32 width, u32 height)
{
	return new handle_t{ width, height, hwnd };
}

inline bool IsValidWindowHandle(WindowHandle handle)
{
	return handle && handle->win && IsWindow(handle->win);
}

inline void ReleaseWindowHandle(WindowHandle& handle)
{
	if (IsValidWindowHandle(handle))
	{
		delete handle;
		handle = nullptr;
	}
}

#elif defined(__linux__)

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
struct handle_t
{
	u32 width;
	u32 height;
	i32 ds;
	Display* dpy;
	Window win;
};
using WindowHandle = struct handle_t*;

inline WindowHandle CreateWindowHandle(u32 width, u32 height, i32 ds, Display* dpy, Window win)
{
	return new handle_t{ width, height, ds, dpy, win };
}

inline bool IsValidWindowHandle(WindowHandle handle)
{
	return handle && handle->win;
}

inline void ReleaseWindowHandle(WindowHandle& handle)
{
	if (IsValidWindowHandle(handle))
	{
		delete handle;
		handle = nullptr;
	}
}

#if !defined(MAX_PATH)
 #define MAX_PATH 260
#endif

#else

#error "only windows and linux are supported at this time!"

#endif
