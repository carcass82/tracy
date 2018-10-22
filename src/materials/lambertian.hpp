/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class Lambertian : public IMaterial
{
public:
    Lambertian(ITexture* a)
        : albedo(a)
    {
    }

    bool scatter(const Ray& r_in, const HitData& rec, ScatterData& srec) const override final
    {
        vec3 target = rec.p + rec.normal + random_on_unit_sphere();
        srec.scattered = Ray(rec.p, normalize(target - rec.p));
        srec.attenuation = albedo->value(rec.uv, rec.p);

        return true;
    }

    vec3 emitted(const Ray& r_in, const HitData& rec, const vec2& uv, const vec3& p) const override final
    {
        return ZERO;
    }

private:
    const vec3 ZERO;
    ITexture* albedo;
};

