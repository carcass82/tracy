/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

inline float fastrand()
{
    static uint32_t s_RndState = 123456789;
#pragma omp threadprivate(s_RndState)
    
    uint32_t x = s_RndState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    s_RndState = x;
    
    return (x & 0xffffff) / 16777216.0f;
}

inline vec3 random_on_unit_sphere()
{
    float z = fastrand() * 2.f - 1.f;
    float a = fastrand() * 2.f * PI;
    float r = sqrtf(max(.0f, 1.f - z * z));

    return vec3{ r * cosf(a), r * sinf(a), z };
}

inline float schlick(float cos, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf((1.f - cos), 5.f);
}

inline vec3 min3(const vec3& a, const vec3& b)
{
    return vec3{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

inline vec3 max3(const vec3& a, const vec3& b)
{
    return vec3{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

inline vec3 sqrtf3(const vec3& a)
{
    return vec3{ sqrtf(a.x), sqrtf(a.y), sqrtf(a.z) };
}

inline vec3 clamp3(const vec3& a, float min, float max)
{
    return vec3{ clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max) };
}
