/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include "ext/cclib/cclib.h"
using cc::math::vec2;
using cc::math::vec3;
using cc::util::clamp;

class texture
{
public:
    virtual vec3 value(const vec2& uv, const vec3& p) const = 0;
};
