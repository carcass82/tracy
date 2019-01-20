/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "material.hpp"

class Metal : public IMaterial
{
public:
    Metal(const vec3& a, float f)
        : albedo(a)
        , roughness(f)
    {
    }

    virtual bool scatter(const Ray& r_in, const HitData& rec, ScatterData& s_rec) const override final
    {
        vec3 reflected = reflect(normalize(r_in.get_direction()), rec.normal);
        s_rec.scattered = Ray(rec.point, reflected + roughness * random_on_unit_sphere());
        s_rec.attenuation = albedo;

        return true;
    }

    virtual vec3 emitted(const Ray& r_in, const HitData& rec, const vec2& uv, const vec3& p) const override final
    {
        return ZERO;
    }

private:
    const vec3 ZERO{.0f, .0f, .0f};
    vec3 albedo;
    float roughness;
};
