/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "shape.hpp"

class Sphere : public IShape
{
public:
    Sphere()
    {
    }

    Sphere(vec3 c, float r, IMaterial* ptr)
        : center(c)
        , radius(r)
        , radius2(r * r)
        , mat(ptr)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        // see ShapeList
        return false;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const override final
    {
        rec.point = r.point_at(rec.t);
        rec.normal = (rec.point - center) / radius;
        rec.uv = get_uv_at((rec.point - center) / radius);
        rec.material = mat;
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        min = center - radius;
        max = center + radius;
    }

    virtual IMaterial* get_material() const override final
    {
        return mat;
    }

    virtual uint32_t get_id() const override final
    {
        return make_id('S', 'P', 'H');
    }

    friend class ShapeList;

private:
    vec2 get_uv_at(const vec3& p) const
    {
        float phi = atan2f(p.z, p.x);
        float theta = asinf(p.y);

        return vec2{ 1.0f - (phi + PI) / (2.0f * PI), (theta + PI / 2.0f) / PI };
    }

    vec3 center;
    float radius;
    float radius2;
    IMaterial* mat;
};
