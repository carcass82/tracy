#pragma once

#include <cuda_runtime.h>

template<typename T> inline __device__ T min(const T& a, const T& b) { return !(b < a) ? a : b; }
template<typename T> inline __device__ T max(const T& a, const T& b) { return (a < b) ? b : a; }
template<> inline __device__ float min<float>(const float& a, const float& b) { return fmin(a, b); }
template<> inline __device__ float max<float>(const float& a, const float& b) { return fmax(a, b); }
template<> inline __device__ float3 min<float3>(const float3& a, const float3& b) { return make_float3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z)); }
template<> inline __device__ float3 max<float3>(const float3& a, const float3& b) { return make_float3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z)); }

template<typename T, size_t N> static inline __device__ size_t array_size(const T(&)[N]) { return N; }

inline __device__ float3  operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __device__ float3  operator+(float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
inline __device__ float3& operator+=(float3& a, const float3& b)      { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline __device__ float3  operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __device__ float3  operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
inline __device__ float3  operator*(const float a, const float3& b)   { return make_float3(a * b.x, a * b.y, a * b.z); }
inline __device__ float3  operator/(float b, const float3& a) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __device__ float3  operator/(const float3& a, const float b)   { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __device__ float3  operator/(const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }

inline __device__ float dot(const float3& a, const float3& b)         { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float3 cross(const float3& a, const float3& b)      { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline __device__ float length(const float3& v)                       { return sqrtf(dot(v, v)); }
inline __device__ float3 normalize(const float3& v)                   { float invLen = rsqrtf(dot(v, v)); return invLen * v; }
inline __device__ float3 reflect(const float3& i, const float3& n)    { return i - 2.f * dot(i, n) * n; }

__device__ const float PI = 3.1415926535897932f;
inline __device__ float radians(float deg) { return deg * PI / 180.0f; }

__device__ bool refract(const float3& I, const float3& N, float eta, float3& refracted)
{
    float NdotI = dot(N, I);
    float k = 1.f - eta * eta * (1.f - NdotI * NdotI);
    refracted = eta * I - (eta * NdotI + sqrtf(max(.0f, k))) * N;

    return (k >= .0f);
}

