/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include "ext/cclib/cclib.h"
#include "geom.hpp"
#include "ray.hpp"
#include "materials/material.hpp"
#include "textures/texture.hpp"

using cc::util::max;
using cc::math::radians;
using cc::math::PI;

class NOVTABLE IShape
{
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const = 0;
    virtual void get_hit_data(const Ray& r, HitData& rec) const = 0;
};
