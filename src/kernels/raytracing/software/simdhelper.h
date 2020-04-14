#pragma once

#include <intrin.h>

//
// float type and operators
//
using simd_float = __m128;

simd_float make_simd(float a)                                        { return _mm_load1_ps(&a); }
simd_float make_simd(float a, float b, float c, float d)             { return _mm_set_ps(d, c, b, a); }

void get_simd(float* out, simd_float value)                          { _mm_store_ps(out, value); }

static const simd_float simd_ZERO = make_simd(.0f);
static const simd_float simd_ONE = make_simd(1.f);
static const simd_float simd_MINUS_ONE = make_simd(-1.f);

inline simd_float operator-(const simd_float a, const simd_float b)  { return _mm_sub_ps(a, b); }
inline simd_float operator+(const simd_float a, const simd_float b)  { return _mm_add_ps(a, b); }
inline simd_float operator*(const simd_float a, const simd_float b)  { return _mm_mul_ps(a, b); }
inline simd_float operator/(const simd_float a, const simd_float b)  { return _mm_div_ps(a, b); }
inline simd_float operator<(const simd_float a, const simd_float b)  { return _mm_cmplt_ps(a, b); }
inline simd_float operator<=(const simd_float a, const simd_float b) { return _mm_cmple_ps(a, b); }
inline simd_float operator>(const simd_float a, const simd_float b)  { return _mm_cmpgt_ps(a, b); }
inline simd_float operator>=(const simd_float a, const simd_float b) { return _mm_cmpge_ps(a, b); }
inline simd_float operator|(const simd_float a, const simd_float b)  { return _mm_or_ps(a, b); }
inline simd_float operator&(const simd_float a, const simd_float b)  { return _mm_and_ps(a, b); }

inline simd_float rcp(simd_float a)                                  { return simd_ONE / a; /* return _mm_rcp_ps(a); */ }
inline simd_float lerp(simd_float a, simd_float b, simd_float mask)  { return _mm_blendv_ps(a, b, mask); }

//
// vec3 type and operators
//
struct simd_vec3
{
	simd_float x;
	simd_float y;
	simd_float z;

	inline simd_vec3(simd_float in_x, simd_float in_y, simd_float in_z)
		: x(in_x)
		, y(in_y)
		, z(in_z)
	{}

	inline simd_vec3(const vec3& in_value)
		: x(_mm_load1_ps(&in_value.x))
		, y(_mm_load1_ps(&in_value.y))
		, z(_mm_load1_ps(&in_value.z))
	{}

	inline simd_vec3(const vec3& in_v1, const vec3& in_v2, const vec3& in_v3, const vec3& in_v4)
		: x(_mm_set_ps(in_v1.x, in_v2.x, in_v3.x, in_v4.x))
		, y(_mm_set_ps(in_v1.y, in_v2.y, in_v3.y, in_v4.y))
		, z(_mm_set_ps(in_v1.z, in_v2.z, in_v3.z, in_v4.z))
	{}
};

inline simd_vec3 operator-(const simd_vec3& a, const simd_vec3& b)
{
	return simd_vec3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

inline simd_vec3 cross(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MSUBs
	return simd_vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline simd_float dot(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MADDs
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
