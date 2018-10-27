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
    Sphere(vec3 c, float r, IMaterial* m)
        : center(c)
        , radius(r)
        , radius2(r * r)
        , mat(m)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        vec3 oc{ r.get_origin() - center };

        float a = dot(r.get_direction(), r.get_direction());
        float b = dot(oc, r.get_direction());
        float c = dot(oc, oc) - radius2;

        //
        // b > 0     - ray pointing away from sphere
        // c > 0     - ray does not start inside sphere
        // discr < 0 - ray does not hit the sphere
        //
        if (b <= .0f || c <= .0f)
        {
            float discriminant = b * b - a * c;
            if (discriminant > .0f)
            {
                float sq_bac = sqrtf(discriminant);

                float t0 = (-b - sq_bac) / a;
                if (t0 < t_max && t0 > t_min)
                {
                    rec.t = t0;
                    return true;
                }

                float t1 = (-b + sq_bac) / a;
                if (t1 < t_max && t1 > t_min)
                {
                    rec.t = t1;
                    return true;
                }
            }
        }

        return false;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const
    {
        rec.p = r.point_at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.uv = get_uv_at((rec.p - center) / radius);
        rec.mat_ptr = mat;
    }

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

