/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class Emissive : public IMaterial
{
public:
    Emissive(ITexture* a)
        : emit(a)
    {
    }

    bool scatter(const Ray &, const HitData &, ScatterData &) const override final
    {
        return false;
    }

    vec3 emitted(const Ray& r_in, const HitData& rec, const vec2& uv, const vec3& p) const override final
    {
        return emit->sample(uv, p);
    }

private:
    ITexture* emit;
};

