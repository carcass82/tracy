/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/gtc/noise.hpp"

class texture
{
public:
    virtual glm::vec3 value(float u, float v, const glm::vec3& p) const = 0;
};


class constant_texture : public texture
{
public:
    constant_texture() {}
    constant_texture(const glm::vec3& c) : color(c) {}

    virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override
    {
        return color;
    }

private:
    glm::vec3 color;
};


class checker_texture : public texture
{
public:
    checker_texture() {}
    checker_texture(texture* t0, texture* t1): even(t0), odd(t1) {}

    virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override
    {
        float sines = glm::sin(10.0f * p.x) * glm::sin(10.0f * p.y) * sin(10.0f * p.z);
        return (sines < 0.0f)? odd->value(u, v, p) : even->value(u, v, p);
    }

private:
    texture* even;
    texture* odd;
};

class noise_texture : public texture
{
public:
    noise_texture() {}
    noise_texture(float sc) : scale(sc) {}

    virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override
    {
        return glm::vec3(1.0f) * 0.5f * (1.0f + glm::sin(scale * p.z + 10 * glm::perlin(p)));
    }

private:

    float scale;
};


class image_texture : public texture
{
public:
    image_texture() {}
    image_texture(unsigned char* pixels, int A, int B)
        : data(pixels), nx(A), ny(B) {}

    virtual glm::vec3 value(float u, float v, const glm::vec3& p) const override
    {
        int i = static_cast<int>((u) * nx);
        int j = static_cast<int>((1 - v) * ny - 0.001f);
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i > nx - 1) i = nx - 1;
        if (j > ny - 1) j = ny - 1;

        float r = int(data[3 * i + 3 * nx * j + 0]) / 255.0f;
        float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0f;
        float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0f;

        return glm::vec3(r, g, b);
    }

private:
    unsigned char* data;
    int nx;
    int ny;
};
