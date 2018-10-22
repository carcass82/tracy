/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include <vector>
#include "ext/cclib/cclib.h"
using cc::math::PI;
using cc::math::vec3;
using cc::math::mat3;
using cc::math::length2;
using cc::util::max;

float fastrand()
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

vec3 random_on_unit_sphere()
{
    float z = fastrand() * 2.f - 1.f;
    float a = fastrand() * 2.f * PI;
    float r = sqrtf(max(.0f, 1.f - z * z));

    return vec3{ r * cosf(a), r * sinf(a), z };
}

vec3 random_in_unit_disk()
{
    vec3 p;
    
    do
    {
        p = vec3(fastrand(), fastrand(), 0.0f) * 2.0f - vec3(1.0f, 1.0f, 0.0f);
    } while (length2(p) >= 1.0f);

    return p;
}

float schlick(float cos, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 *= r0;
    return r0 + (1.0f - r0) * pow((1 - cos), 5);
}
