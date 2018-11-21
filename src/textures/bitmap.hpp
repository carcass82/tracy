/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "texture.hpp"

class Bitmap : public ITexture
{
public:
    Bitmap(uint8_t* pixels, int width, int height)
        : data(pixels)
        , nx(width)
        , ny(height)
    {
    }

    virtual vec3 sample(const vec2& uv, const vec3& p) const override final
    {
        int i = clamp(int((uv.s) * nx),                 0, nx - 1);
        int j = clamp(int((1.0f - uv.t) * ny - 0.001f), 0, ny - 1);
        
        float r = data[3 * (i + nx * j) + 0] / 255.0f;
        float g = data[3 * (i + nx * j) + 1] / 255.0f;
        float b = data[3 * (i + nx * j) + 2] / 255.0f;

        return vec3{ r, g, b };
    }

private:
    uint8_t* data;
    int nx;
    int ny;
};
