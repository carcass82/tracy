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
        int i = clamp(int(uv.s * nx),          0, nx - 1);
        int j = clamp(int((1.0f - uv.t) * ny), 0, ny - 1);

        int idx = 3 * (i + nx * j);
        
        float r = data[idx + 0];
        float g = data[idx + 1];
        float b = data[idx + 2];

        return vec3{ r, g, b } / 255.f;
    }

private:
    uint8_t* data;
    int nx;
    int ny;
};
