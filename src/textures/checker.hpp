/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "texture.hpp"

class checker_texture : public Texture
{
public:
    checker_texture() {}
    checker_texture(Texture* t0, Texture* t1): even(t0), odd(t1) {}

    virtual vec3 value(const vec2& uv, const vec3& p) const override
    {
        float sines = sinf(10.0f * p.x) * sinf(10.0f * p.y) * sinf(10.0f * p.z);
        return (sines < 0.0f)? odd->value(uv, p) : even->value(uv, p);
    }

private:
    Texture* even;
    Texture* odd;
};

