/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec2;
using vmath::vec3;
using vutil::clamp;

class texture
{
public:
    virtual vec3 value(const vec2& uv, const vec3& p) const = 0;
};
