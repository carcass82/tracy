/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <cuda_runtime.h>

inline __device__ __host__ float3 min(const float3& a, const float3& b) { return float3{ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z) }; }
inline __device__ __host__ float3 max(const float3& a, const float3& b) { return float3{ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z) }; }

constexpr inline __device__ __host__ float pi()               { return 3.1415926535897932f; }
constexpr inline __device__ __host__ float radians(float deg) { return deg * pi() / 180.0f; }

template<typename T, size_t N> constexpr inline __device__ size_t array_size(const T(&)[N]) { return N; }
template<typename T> inline __device__ void swap(T& a, T& b)                                { T tmp(a); a = b; b = tmp; }

__host__ __device__ constexpr inline uint32_t make_id(char a, char b, char c = '\0', char d = '\0')
{
    return a | b << 8 | c << 16 | d << 24;
}

struct mat4
{
	float4 m[4];
	float nan = NAN;

	constexpr __device__ __host__ float4& operator[](size_t i)                { return m[i]; }
	constexpr __device__ __host__ const float4& operator[](size_t i) const    { return m[i]; }

	constexpr __device__ __host__ const float& operator()(size_t i, size_t j) const { switch (j) { case 0: return m[i].x; case 1: return m[i].y; case 2: return m[i].z; case 3: return m[i].w; default: return nan; } }
};

constexpr inline __device__ __host__ float4  operator+(const float4& a, const float4& b) { return float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
constexpr inline __device__ __host__ float4  operator-(const float4& a, const float4& b) { return float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
constexpr inline __device__ __host__ float4  operator*(const float4& b, const float a)   { return float4{ a * b.x, a * b.y, a * b.z, a * b.w }; }
constexpr inline __device__ __host__ float4  operator*(const float4& a, const float4& b) { return float4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
constexpr inline __device__ __host__ float4& operator/=(float4& a, const float b)        { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }

constexpr inline __device__ __host__ float3  operator+(const float3& a, const float3& b) { return float3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
constexpr inline __device__ __host__ float3  operator+(float a, const float3 & b)        { return float3{ a + b.x, a + b.y, a + b.z }; }
constexpr inline __device__ __host__ float3& operator+=(float3& a, const float3& b)      { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
constexpr inline __device__ __host__ float3  operator-(const float3& a, const float3& b) { return float3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
constexpr inline __device__ __host__ float3  operator-(const float3& b, float a)         { return float3{ b.x - a, b.y - a, b.z - a }; }
constexpr inline __device__ __host__ float3  operator*(const float3& a, const float3& b) { return float3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
constexpr inline __device__ __host__ float3  operator*(const float3& b, const float a)   { return float3{ a * b.x, a * b.y, a * b.z }; }
constexpr inline __device__ __host__ float3  operator*(const float a, const float3& b)   { return float3{ a * b.x, a * b.y, a * b.z }; }
constexpr inline __device__ __host__ float3& operator*=(float3& a, const float3& b)      { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
constexpr inline __device__ __host__ float3  operator/(float b, const float3& a)         { return float3{ b / a.x, b / a.y, b / a.z }; }
constexpr inline __device__ __host__ float3  operator/(const float3& a, const float b)   { return float3{ a.x / b, a.y / b, a.z / b }; }
constexpr inline __device__ __host__ float3  operator/(const float3& a, const float3& b) { return float3{ a.x / b.x, a.y / b.y, a.z / b.z }; }

constexpr inline __device__ __host__ float2  operator+(const float2& a, const float2& b) { return float2{ a.x + b.x, a.y + b.y }; }
constexpr inline __device__ __host__ float2  operator*(const float a, const float2& b)   { return float2{ a * b.x, a * b.y }; }

constexpr inline __device__ __host__ mat4 operator*(const mat4& a, const mat4& b)
{
	return mat4
	{
		float4{ a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z + a[3] * b[0].w },
		float4{ a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z + a[3] * b[1].w },
		float4{ a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z + a[3] * b[2].w },
		float4{ a[0] * b[3].x + a[1] * b[3].y + a[2] * b[3].z + a[3] * b[3].w }
	};
}

constexpr inline __device__ __host__ float4 operator*(const mat4& a, const float4& b)
{
	return float4
	{
		b.x * a[0].x + b.y * a[1].x + b.z * a[2].x + b.w * a[3].x,
		b.x * a[0].y + b.y * a[1].y + b.z * a[2].y + b.w * a[3].y,
		b.x * a[0].z + b.y * a[1].z + b.z * a[2].z + b.w * a[3].z,
		b.x * a[0].w + b.y * a[1].w + b.z * a[2].w + b.w * a[3].w
	};
}

inline __device__ __host__ float dot(const float3& a, const float3& b)                   { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ __host__ float3 cross(const float3& a, const float3& b)                { return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline __device__ __host__ float length(const float3& v)                                 { return sqrtf(dot(v, v)); }
inline __device__ __host__ float3 normalize(const float3& v)                             { return rsqrtf(dot(v, v)) * v; }
inline __device__ __host__ float3 reflect(const float3& i, const float3& n)              { return i - 2.f * n * dot(i, n); }

inline __device__ __host__ bool refract(const float3& I, const float3& N, float eta, float3& refracted)
{
    float NdotI = dot(N, I);
    float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

    refracted = eta * I - (eta * NdotI + sqrtf(fmaxf(.0f, k))) * N;
    
    return (k > .0f);
}

inline __device__ __host__ mat4 lookAt(const float3& eye, const float3& center, const float3& up)
{
	const float3 f(normalize(center - eye));
	const float3 s(normalize(cross(f, up)));
	const float3 u(cross(s, f));

	return mat4
	{
		float4{          s.x,          u.x,          -f.x,  0.0f },
		float4{          s.y,          u.y,          -f.y,  0.0f },
		float4{          s.z,          u.z,          -f.z,  0.0f },
		float4{ -dot(s, eye), -dot(u, eye),   dot(f, eye),  1.0f }
	};
}

inline __device__ __host__ mat4 perspective(float fovy, float aspect, float znear, float zfar)
{
	const float f = 1.f / tanf(fovy / 2.0f);

	return mat4
	{
		float4{ f / aspect,  0.0f,                                    0.0f,   0.0f },
		float4{       0.0f,     f,                                    0.0f,   0.0f },
		float4{       0.0f,  0.0f,        -(zfar + znear) / (zfar - znear),  -1.0f },
		float4{       0.0f,  0.0f,  -(2.f * zfar * znear) / (zfar - znear),   0.0f }
	};
}

constexpr inline __device__ __host__ mat4 inverse(const mat4& m)
{
	float coef00 = m(2, 2) * m(3, 3) - m(3, 2) * m(2, 3);
	float coef02 = m(1, 2) * m(3, 3) - m(3, 2) * m(1, 3);
	float coef03 = m(1, 2) * m(2, 3) - m(2, 2) * m(1, 3);

	float coef04 = m(2, 1) * m(3, 3) - m(3, 1) * m(2, 3);
	float coef06 = m(1, 1) * m(3, 3) - m(3, 1) * m(1, 3);
	float coef07 = m(1, 1) * m(2, 3) - m(2, 1) * m(1, 3);

	float coef08 = m(2, 1) * m(3, 2) - m(3, 1) * m(2, 2);
	float coef10 = m(1, 1) * m(3, 2) - m(3, 1) * m(1, 2);
	float coef11 = m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2);

	float coef12 = m(2, 0) * m(3, 3) - m(3, 0) * m(2, 3);
	float coef14 = m(1, 0) * m(3, 3) - m(3, 0) * m(1, 3);
	float coef15 = m(1, 0) * m(2, 3) - m(2, 0) * m(1, 3);

	float coef16 = m(2, 0) * m(3, 2) - m(3, 0) * m(2, 2);
	float coef18 = m(1, 0) * m(3, 2) - m(3, 0) * m(1, 2);
	float coef19 = m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2);

	float coef20 = m(2, 0) * m(3, 1) - m(3, 0) * m(2, 1);
	float coef22 = m(1, 0) * m(3, 1) - m(3, 0) * m(1, 1);
	float coef23 = m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1);

	float4 fac0{ coef00, coef00, coef02, coef03 };
	float4 fac1{ coef04, coef04, coef06, coef07 };
	float4 fac2{ coef08, coef08, coef10, coef11 };
	float4 fac3{ coef12, coef12, coef14, coef15 };
	float4 fac4{ coef16, coef16, coef18, coef19 };
	float4 fac5{ coef20, coef20, coef22, coef23 };

	float4 vec0{ m(1, 0), m(0, 0), m(0, 0), m(0, 0) };
	float4 vec1{ m(1, 1), m(0, 1), m(0, 1), m(0, 1) };
	float4 vec2{ m(1, 2), m(0, 2), m(0, 2), m(0, 2) };
	float4 vec3{ m(1, 3), m(0, 3), m(0, 3), m(0, 3) };

	float4 inv0{ vec1 * fac0 - vec2 * fac1 + vec3 * fac2 };
	float4 inv1{ vec0 * fac0 - vec2 * fac3 + vec3 * fac4 };
	float4 inv2{ vec0 * fac1 - vec1 * fac3 + vec3 * fac5 };
	float4 inv3{ vec0 * fac2 - vec1 * fac4 + vec2 * fac5 };

	float4 sign_a{ +1, -1, +1, -1 };
	float4 sign_b{ -1, +1, -1, +1 };

	mat4 inv{ float4{ inv0 * sign_a }, float4{ inv1 * sign_b }, float4{ inv2 * sign_a }, float4{ inv3 * sign_b } };

	float4 row0{ inv(0, 0), inv(1, 0), inv(2, 0), inv(3, 0) };

	float4 dot0(m[0] * row0);
	float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);
	float one_over_det = 1.f / dot1;

	return mat4{ inv[0] * one_over_det, inv[1] * one_over_det, inv[2] * one_over_det, inv[3] * one_over_det };
}
