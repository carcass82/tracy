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
    Ray(const vec3& a, const vec3& b) : origin_(a), direction_(b) {}

    constexpr inline const vec3& GetOrigin() const noexcept    { return origin_;    }
    constexpr inline const vec3& GetDirection() const noexcept { return direction_; }
    constexpr inline vec3 PointAt(float t) const noexcept      { return origin_ + t * direction_; }

private:
    vec3 origin_;
    vec3 direction_;
};
