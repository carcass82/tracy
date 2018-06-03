/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include "ext/cclib/cclib.h"
using cc::math::vec3;
using cc::math::mat3;

//
// probability density function
//
class NOVTABLE pdf
{
public:
    virtual float value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};
