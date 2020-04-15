#pragma once

#if defined(_MSC_VER)
 #include <intrin.h>
#else
 #include <x86intrin.h>
#endif

#if SIMD_SSE

#define SIMD_WIDTH 4

//
// float type and operators
//
struct simd_float
{
	__m128 v;
	alignas(16) float float_v[4];

	inline simd_float() noexcept                                             {}
	inline simd_float(float in_v) noexcept                                   { v = _mm_load1_ps(&in_v); }
	inline simd_float(float in0, float in1, float in2, float in3) noexcept   { v = _mm_set_ps(in3, in2, in1, in0); }
	inline simd_float(const float in_v[4]) noexcept                          { v = _mm_loadu_ps(in_v); }
	inline simd_float(__m128 in_v) noexcept : v(in_v)                        {}

	inline float* get_float()
	{
		_mm_store_ps(float_v, v);
		return float_v;
	}
};

static const simd_float simd_ZERO(.0f);
static const simd_float simd_ONE(1.f);
static const simd_float simd_MINUS_ONE(-1.f);

inline simd_float operator-(const simd_float a, const simd_float b)  { return _mm_sub_ps(a.v, b.v); }
inline simd_float operator+(const simd_float a, const simd_float b)  { return _mm_add_ps(a.v, b.v); }
inline simd_float operator*(const simd_float a, const simd_float b)  { return _mm_mul_ps(a.v, b.v); }
inline simd_float operator/(const simd_float a, const simd_float b)  { return _mm_div_ps(a.v, b.v); }
inline simd_float operator<(const simd_float a, const simd_float b)  { return _mm_cmplt_ps(a.v, b.v); }
inline simd_float operator<=(const simd_float a, const simd_float b) { return _mm_cmple_ps(a.v, b.v); }
inline simd_float operator>(const simd_float a, const simd_float b)  { return _mm_cmpgt_ps(a.v, b.v); }
inline simd_float operator>=(const simd_float a, const simd_float b) { return _mm_cmpge_ps(a.v, b.v); }
inline simd_float operator|(const simd_float a, const simd_float b)  { return _mm_or_ps(a.v, b.v); }
inline simd_float operator&(const simd_float a, const simd_float b)  { return _mm_and_ps(a.v, b.v); }

inline simd_float rcp(simd_float a)                                  { return _mm_rcp_ps(a.v); }
inline simd_float lerp(simd_float a, simd_float b, simd_float mask)  { return _mm_blendv_ps(a.v, b.v, mask.v); }

//
// vec3 type and operators
//
struct simd_vec3
{
	simd_float x;
	simd_float y;
	simd_float z;

	inline simd_vec3() noexcept
	{}

	inline simd_vec3(simd_float in_x, simd_float in_y, simd_float in_z) noexcept
		: x(in_x)
		, y(in_y)
		, z(in_z)
	{}

	inline simd_vec3(const vec3& in_value) noexcept
		: x(in_value.x)
		, y(in_value.y)
		, z(in_value.z)
	{}

	inline simd_vec3(const vec3 in_v[4]) noexcept
		: x(in_v[0].x, in_v[1].x, in_v[2].x, in_v[3].x)
		, y(in_v[0].y, in_v[1].y, in_v[2].y, in_v[3].y)
		, z(in_v[0].z, in_v[1].z, in_v[2].z, in_v[3].z)
	{}
};

inline simd_vec3 operator-(const simd_vec3& a, const simd_vec3& b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline simd_vec3 cross(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MSUBs
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline simd_float dot(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MADDs
	return a.x * b.x + a.y * b.y + a.z * b.z;
}



#elif SIMD_AVX

#define SIMD_WIDTH 8

//
// float type and operators
//
struct simd_float
{
	__m256 v;
	alignas(32) float float_v[8];

	inline simd_float() noexcept {}
	
	inline simd_float(float in_v) noexcept
	{
		v = _mm256_set1_ps(in_v);
	}
	
	inline simd_float(float in0, float in1, float in2, float in3, float in4, float in5, float in6, float in7) noexcept
	{
		v = _mm256_set_ps(in7, in6, in5, in4, in3, in2, in1, in0);
	}
	
	inline simd_float(const float in_v[8]) noexcept { v = _mm256_loadu_ps(in_v); }
	
	inline simd_float(__m256 in_v) noexcept : v(in_v) {}

	inline float* get_float()
	{
		_mm256_store_ps(float_v, v);
		return float_v;
	}
};

static const simd_float simd_ZERO(.0f);
static const simd_float simd_ONE(1.f);
static const simd_float simd_MINUS_ONE(-1.f);

inline simd_float operator-(const simd_float a, const simd_float b)  { return _mm256_sub_ps(a.v, b.v); }
inline simd_float operator+(const simd_float a, const simd_float b)  { return _mm256_add_ps(a.v, b.v); }
inline simd_float operator*(const simd_float a, const simd_float b)  { return _mm256_mul_ps(a.v, b.v); }
inline simd_float operator/(const simd_float a, const simd_float b)  { return _mm256_div_ps(a.v, b.v); }
inline simd_float operator<(const simd_float a, const simd_float b)  { return _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ); }
inline simd_float operator<=(const simd_float a, const simd_float b) { return _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ); }
inline simd_float operator>(const simd_float a, const simd_float b)  { return _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ); }
inline simd_float operator>=(const simd_float a, const simd_float b) { return _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ); }
inline simd_float operator|(const simd_float a, const simd_float b)  { return _mm256_or_ps(a.v, b.v); }
inline simd_float operator&(const simd_float a, const simd_float b)  { return _mm256_and_ps(a.v, b.v); }

inline simd_float rcp(simd_float a)                                  { return _mm256_rcp_ps(a.v); }
inline simd_float lerp(simd_float a, simd_float b, simd_float mask)  { return _mm256_blendv_ps(a.v, b.v, mask.v); }

//
// vec3 type and operators
//
struct simd_vec3
{
	simd_float x;
	simd_float y;
	simd_float z;

	inline simd_vec3() noexcept
	{}

	inline simd_vec3(simd_float in_x, simd_float in_y, simd_float in_z) noexcept
		: x(in_x)
		, y(in_y)
		, z(in_z)
	{}

	inline simd_vec3(const vec3& in_value) noexcept
		: x(in_value.x)
		, y(in_value.y)
		, z(in_value.z)
	{}

	inline simd_vec3(const vec3 in_v[8]) noexcept
		: x(in_v[0].x, in_v[1].x, in_v[2].x, in_v[3].x, in_v[4].x, in_v[5].x, in_v[6].x, in_v[7].x)
		, y(in_v[0].y, in_v[1].y, in_v[2].y, in_v[3].y, in_v[4].y, in_v[5].y, in_v[6].y, in_v[7].y)
		, z(in_v[0].z, in_v[1].z, in_v[2].z, in_v[3].z, in_v[4].z, in_v[5].z, in_v[6].z, in_v[7].z)
	{}
};

inline simd_vec3 operator-(const simd_vec3& a, const simd_vec3& b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline simd_vec3 cross(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MSUBs
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline simd_float dot(const simd_vec3& a, const simd_vec3& b)
{
	// TODO: use MADDs
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

#else

#error "no SIMD instructions selected"

#endif
