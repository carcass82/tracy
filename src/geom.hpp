/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <vector>
#include <algorithm>
#include "tmath.h"
using vmath::vec3;
using vmath::length2;

float fastrand()
{
    static uint32_t s_seed = 123456789;

    s_seed = (214013 * s_seed + 2531011);
    return ((s_seed >> 16) & 0x7FFF) / 32768.0f;
}

vec3 random_in_unit_sphere()
{
    vec3 p;
    do
        p = vec3(fastrand(), fastrand(), fastrand()) * 2.0f - vec3(1.0f, 1.0f, 1.0f);
    while (length2(p) >= 1.0f);

    return p;
}

vec3 random_in_unit_disk()
{
    vec3 p;
    do
        p = vec3(fastrand(), fastrand(), 0.0f) * 2.0f - vec3(1.0f, 1.0f, 0.0f);
    while (length2(p) >= 1.0f);

    return p;
}

float schlick(float cos, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 *= r0;

    return r0 + (1.0f - r0) * pow((1 - cos), 5);
}
