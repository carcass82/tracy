/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

//
// PRNG from
// https://en.wikipedia.org/wiki/Xorshift
// (Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs")
//
//inline float fastrand()
//{
//    static uint32_t s_RndState = 123456789;
//    #pragma omp threadprivate(s_RndState)
//
//    uint32_t x = s_RndState;
//    x ^= x << 13;
//    x ^= x >> 17;
//    x ^= x << 5;
//    s_RndState = x;
//
//    return (x & 0xffffff) / 16777216.0f;
//}

//
// PRNG from
// https://en.wikipedia.org/wiki/Permuted_congruential_generator
//
//inline float fastrand()
//{
//    static uint64_t mcg_state = 0xcafef00dd15ea5e4u;
//    #pragma omp threadprivate(mcg_state)
//    
//    uint64_t x = mcg_state;
//    unsigned count = (unsigned)(x >> 61);
//    mcg_state = x * 6364136223846793005u;
//    x ^= x >> 22;
//    
//    return ((uint32_t)(x >> (22 + count)) & 0xffffff) / 16777216.f;
//}

//
// PRNG from
// https://software.intel.com/en-us/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor/
//
inline float fastrand()
{
    static uint32_t g_state = 0xdeadbeef;
    #pragma omp threadprivate(g_state)

    g_state = (214013u * g_state + 2531011u);
    return (g_state & 0xffffff) / 16777216.f;
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
