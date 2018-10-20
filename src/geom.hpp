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
    static uint32_t s_RndState = 1;
    
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

vec3 random_in_unit_sphere()
{
    vec3 p;
    
    do
    {
        p = vec3(fastrand(), fastrand(), fastrand()) * 2.0f - vec3(1.0f, 1.0f, 1.0f);
    } while (length2(p) >= 1.0f);

    return p;
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

vec3 random_cosine_GetDirection()
{
    float r1 = fastrand();
    float r2 = fastrand();
    float phi = PI * 2.f * r1;

    return vec3{ cosf(phi) * 2.f * sqrtf(r2), sinf(phi) * 2.f * sqrtf(r2), sqrtf(1.f - r2) };
}

vec3 random_to_sphere(float radius, float distance2)
{
    float r1 = fastrand();
    float r2 = fastrand();
    float phi = PI * 2.f * r1;

    float z = 1.f + r2 * (sqrtf(1.f - radius * radius / distance2) - 1.f);
    float x = cosf(phi) * sqrtf(1.f - z * z);
    float y = sinf(phi) * sqrtf(1.f - z * z);

    return vec3{ x, y, z };
}

mat3 build_orthonormal_basis(const vec3& w)
{
    vec3 axis = (fabsf(w.x) > .9f)? vec3{0, 1, 0} : vec3{1, 0, 0};

    mat3 res;
    res[2] = normalize(w);
    res[1] = normalize(cross(res[2], axis));
    res[0] = normalize(cross(res[2], res[1]));

    return res;
}

float schlick(float cos, float ref_idx)
{
    float r0 = powf((1.0f - ref_idx) / (1.0f + ref_idx), 2.0f);
    return r0 + (1.0f - r0) * pow((1 - cos), 5);
}
