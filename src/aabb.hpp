/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once
#include "tmath.h"
#include "ray.hpp"
using vutil::min;
using vutil::max;
using vmath::vec3;

//
// simple AABB
//
struct aabb
{
    vec3 vmin;
    vec3 vmax;


    aabb() : vmin() , vmax() {}
    aabb(const vec3& a, const vec3& b) : vmin(a) , vmax(b) {}

    vec3 center() const { return (vmax + vmin) / 2.f; }

    void expand(const aabb& other_box)
    {
        vmin = { min(vmin.x, other_box.vmin.x), min(vmin.y, other_box.vmin.y), min(vmin.z, other_box.vmin.z) };
        vmax = { max(vmax.x, other_box.vmax.x), max(vmax.y, other_box.vmax.y), max(vmax.z, other_box.vmax.z) };
    }

    float distance(const vec3& p) const
    {
        float res = .0f;

        if (p.x < vmin.x) res += (vmin.x - p.x) * (vmin.x - p.x);
        if (p.x > vmax.x) res += (p.x - vmax.x) * (p.x - vmax.x);
        if (p.y < vmin.y) res += (vmin.y - p.y) * (vmin.y - p.y);
        if (p.y > vmax.y) res += (p.y - vmax.y) * (p.y - vmax.y);
        if (p.z < vmin.z) res += (vmin.z - p.z) * (vmin.z - p.z);
        if (p.z > vmax.z) res += (p.z - vmax.z) * (p.z - vmax.z);

        return res;
    }

    bool hit(const Ray& r, float tmin, float tmax) const
    {
        vec3 a = (vmin - r.origin()) / r.direction();
        vec3 b = (vmax - r.origin()) / r.direction();

        return (!(min(max(a.x, b.x), tmax) <= max(min(a.x, b.x), tmin)) ||
                 (min(max(a.y, b.y), tmax) <= max(min(a.y, b.y), tmin)) ||
                 (min(max(a.z, b.z), tmax) <= max(min(a.z, b.z), tmin)));
    }
};
