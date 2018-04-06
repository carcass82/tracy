/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec3;
using vutil::clamp;

class texture
{
public:
    virtual vec3 value(float u, float v, const vec3& p) const = 0;
};


class constant_texture : public texture
{
public:
    constant_texture() {}
    constant_texture(const vec3& c) : color(c) {}

    virtual vec3 value(float u, float v, const vec3& p) const override
    {
        return color;
    }

private:
    vec3 color;
};


class checker_texture : public texture
{
public:
    checker_texture() {}
    checker_texture(texture* t0, texture* t1): even(t0), odd(t1) {}

    virtual vec3 value(float u, float v, const vec3& p) const override
    {
        float sines = sinf(10.0f * p.x) * sinf(10.0f * p.y) * sinf(10.0f * p.z);
        return (sines < 0.0f)? odd->value(u, v, p) : even->value(u, v, p);
    }

private:
    texture* even;
    texture* odd;
};


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

    virtual vec3 value(float u, float v, const vec3& p) const override
    {
        int i = clamp(static_cast<int>((u) * nx),                 0, nx - 1);
        int j = clamp(static_cast<int>((1.0f - v) * ny - 0.001f), 0, ny - 1);

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
