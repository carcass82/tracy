#pragma once

template<typename T, size_t N> static inline __device__ size_t array_size(const T(&)[N]) { return N; }

static inline __device__ float3  operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline __device__ float3& operator+=(float3& a, const float3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static inline __device__ float3  operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline __device__ float3  operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline __device__ float3  operator*(const float a, const float3& b) { return make_float3(a * b.x, a * b.y, a * b.z); }
static inline __device__ float3  operator/(const float3& a, const float b) { return make_float3(a.x / b, a.y / b, a.z / b); }

static inline __device__ float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline __device__ float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline __device__ float length(const float3& v) { return sqrtf(dot(v, v)); }
static inline __device__ float3 normalize(const float3& v) { float invLen = rsqrtf(dot(v, v)); return invLen * v; }
static inline __device__ float3 reflect(const float3& i, const float3& n) { return i - 2.f * dot(i, n) * n; }

__device__ const float PI = 3.1415926535897932f;
static inline __device__ float radians(float deg) { return deg * PI / 180.0f; }

__device__ bool refract(const float3& I, const float3& N, float eta, float3& refracted)
{
    float NdotI = dot(N, I);
    float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

    if (k >= .0f)
    {
        refracted = float3(eta * I - (eta * NdotI + sqrtf(k)) * N);
        return true;
    }
    else
    {
        return false;
    }
}

