/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec3;

//
// a ray represented in its parametric form
// A - ray origin
// B - ray direction
//
struct ray
{
    vec3 A;
    vec3 B;

    ray() {}
    ray(const vec3& a, const vec3& b) : A(a), B(b) {}

    const vec3& origin() const    { return A; }
    const vec3& direction() const { return B; }
    vec3 pt(float t) const        { return A + t * B; }
};
