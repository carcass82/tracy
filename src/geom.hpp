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

float fastrand()
{
    static uint32_t s_seed = 123456789;

    s_seed = (214013 * s_seed + 2531011);
    return ((s_seed >> 16) & 0x7FFF) / 32768.0f;
}

vec3 random_in_unit_sphere()
{
    vec3 p;
    do {
        p = vec3(fastrand(), fastrand(), fastrand()) * 2.0f - vec3(1.0f, 1.0f, 1.0f);
    } while (length2(p) >= 1.0f);

    return p;
}

vec3 random_on_unit_sphere()
{
    vec3 p;
    do {
        p = vec3(fastrand(), fastrand(), fastrand()) * 2.0f - vec3(1.0f, 1.0f, 1.0f);
    } while (length2(p) >= 1.0f);

    return normalize(p);
}

vec3 random_in_unit_disk()
{
    vec3 p;
    do {
        p = vec3(fastrand(), fastrand(), 0.0f) * 2.0f - vec3(1.0f, 1.0f, 0.0f);
    } while (length2(p) >= 1.0f);

    return p;
}

vec3 random_cosine_direction()
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
