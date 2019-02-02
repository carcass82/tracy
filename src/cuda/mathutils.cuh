/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <cuda_runtime.h>

template<typename T> inline __device__ __host__ T min(const T& a, const T& b)              { return !(b < a) ? a : b; }
template<typename T> inline __device__ __host__ T max(const T& a, const T& b)              { return (a < b) ? b : a; }
template<>           inline __device__ __host__ float min<float>(const float& a, const float& b)     { return fminf(a, b); }
template<>           inline __device__ __host__ float max<float>(const float& a, const float& b)     { return fmaxf(a, b); }
template<>           inline __device__ __host__ float3 min<float3>(const float3& a, const float3& b) { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
template<>           inline __device__ __host__ float3 max<float3>(const float3& a, const float3& b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }

__device__ const float PI = 3.1415926535897932f;
inline __device__ __host__ float radians(float deg) { return deg * 3.1415926535897932f / 180.0f; }

template<typename T, size_t N> inline __device__ size_t array_size(const T(&)[N]) { return N; }
template<typename T> inline __device__ void swap(T& a, T& b) { T tmp(a); a = b; b = tmp; }

__host__ __device__ constexpr inline uint32_t make_id(char a, char b, char c = '\0', char d = '\0')
{
    return a | b << 8 | c << 16 | d << 24;
}

inline __device__ __host__ float3  operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ __host__ float3  operator+(float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
inline __device__ __host__ float3& operator+=(float3& a, const float3& b)      { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline __device__ __host__ float3  operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ __host__ float3  operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __device__ __host__ float3  operator*(const float3& b, const float a)   { return make_float3(a * b.x, a * b.y, a * b.z); }
inline __device__ __host__ float3  operator*(const float a, const float3& b)   { return make_float3(a * b.x, a * b.y, a * b.z); }
inline __device__ __host__ float3& operator*=(float3& a, const float3& b)      { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
inline __device__ __host__ float3  operator/(float b, const float3& a)         { return make_float3(b / a.x, b / a.y, b / a.z); }
inline __device__ __host__ float3  operator/(const float3& a, const float b)   { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __device__ __host__ float3  operator/(const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }

inline __device__ __host__ float2  operator+(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ __host__ float2  operator*(const float a, const float2& b)   { return make_float2(a * b.x, a * b.y); }

inline __device__ __host__ float dot(const float3& a, const float3& b)         { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ __host__ float3 cross(const float3& a, const float3& b)      { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline __device__ __host__ float length(const float3& v)                       { return sqrtf(dot(v, v)); }
inline __device__ __host__ float3 normalize(const float3& v)                   { return rsqrtf(dot(v, v)) * v; }
inline __device__ __host__ float3 reflect(const float3& i, const float3& n)    { return i - 2.f * n * dot(i, n); }

inline __device__ __host__ bool refract(const float3& I, const float3& N, float eta, float3& refracted)
{
    float NdotI = dot(N, I);
    float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

    refracted = eta * I - (eta * NdotI + sqrtf(fmaxf(.0f, k))) * N;
    
    return (k > .0f);
}
