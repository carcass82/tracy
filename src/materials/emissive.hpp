/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class emissive : public material
{
public:
    emissive(Texture* a)
        : emit(a)
    {
    }

    virtual vec3 emitted(const Ray& r_in, const hit_record& rec, const vec2& uv, const vec3& p) const override
    {
        if (dot(rec.normal, r_in.GetDirection()) < .0f)
        {
            return emit->value(uv, p);
        }
        else
        {
            return vec3();
        }
    }

private:
    Texture* emit;
};

