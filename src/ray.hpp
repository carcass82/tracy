/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include "ext/cclib/cclib.h"
using cc::math::vec3;

//
// a ray represented in its parametric form
// A - ray origin
// B - ray direction
//
class Ray
{
public:
    Ray() {}
    Ray(const vec3& a, const vec3& b) : A(a), B(b) {}

    const vec3& origin() const    { return A; }
    const vec3& direction() const { return B; }
    vec3 pt(float t) const        { return A + t * B; }

private:
    vec3 A;
    vec3 B;
};
