/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "texture.hpp"

class Constant : public ITexture
{
public:
    Constant()
    {
    }

    Constant(const vec3& c)
        : color(c)
    {
    }

    virtual vec3 value(const vec2& uv, const vec3& p) const override
    {
        return color;
    }

private:
    vec3 color;
};

