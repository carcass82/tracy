/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

#include <iostream>
#include "ext/cclib/cclib.h"
#include "geom.hpp"
#include "ray.hpp"
#include "aabb.hpp"
#include "material.hpp"
#include "texture.hpp"

using cc::util::max;
using cc::math::radians;
using cc::math::PI;

class material;
class isotropic;

class hitable
{
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
    virtual float pdf_value(const vec3& o, const vec3& v) const { return 0.f; }
    virtual vec3 random(const vec3& o) const { return vec3{ 1, 0, 0 }; }
};
