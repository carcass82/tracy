/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

class NOVTABLE IShape
{
public:
    virtual ~IShape() {}

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const = 0;
    virtual void get_hit_data(const Ray& r, HitData& rec) const = 0;
    virtual void get_bounds(vec3& min, vec3& max) const = 0;
};
