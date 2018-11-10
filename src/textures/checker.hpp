/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "texture.hpp"

class Checker : public ITexture
{
public:
    Checker()
    {
    }

    Checker(ITexture* t0, ITexture* t1)
        : even(t0)
        , odd(t1)
    {
    }

    virtual vec3 sample(const vec2& uv, const vec3& p) const override final
    {
        float sines = sinf(10.0f * p.x) * sinf(10.0f * p.y) * sinf(10.0f * p.z);
        return (sines < 0.0f) ? odd->sample(uv, p) : even->sample(uv, p);
    }
    
private:
    ITexture* even;
    ITexture* odd;
};
