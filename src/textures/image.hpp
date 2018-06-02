#pragma once
#include "texture.hpp"

class image_texture : public texture
{
public:
    image_texture() {}
    image_texture(uint8_t* pixels, int width, int height)
        : data(pixels)
        , nx(width)
        , ny(height)
    {
    }

    virtual vec3 value(const vec2& uv, const vec3& p) const override
    {
        int i = clamp(static_cast<int>((uv.s) * nx),                 0, nx - 1);
        int j = clamp(static_cast<int>((1.0f - uv.t) * ny - 0.001f), 0, ny - 1);

        float r = data[3 * i + 3 * nx * j + 0] / 255.0f;
        float g = data[3 * i + 3 * nx * j + 1] / 255.0f;
        float b = data[3 * i + 3 * nx * j + 2] / 255.0f;

        return vec3(r, g, b);
    }

private:
    uint8_t* data;
    int nx;
    int ny;
};
