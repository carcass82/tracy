/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "common.h"

class Texture
{
public:
    Texture()
    {}

    Texture(int32_t in_width, int32_t in_height, uint8_t* in_pixels, bool sRGB = false)
        : width{ in_width }, height{ in_height }, valid{ true }
    {
        pixels = new vec4[width * height];

        for (int32_t i = 0; i < width * height; ++i)
        {
            vec4 pixel = vec4(in_pixels[i * bpp], in_pixels[i * bpp + 1], in_pixels[i * bpp + 2], in_pixels[i * bpp + 3]) / 255.f;
            pixels[i] = sRGB ? linear(pixel) : pixel;
        }
    }

    ~Texture()
    {
        delete[] pixels;
    }

    // disable copying
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept
        : width{ other.width }, height{ other.height }, bpp{ other.bpp }, pixels{ std::exchange(other.pixels, nullptr) }, valid{ other.valid }
    {}

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            valid = std::move(other.valid);
            width = std::move(other.width);
            height = std::move(other.height);
            bpp = std::move(other.bpp);
            pixels = std::exchange(other.pixels, nullptr);
        }
        return *this;
    }

    const vec4& GetPixel(const vec2& uv) const
    {
        using cc::math::frac;

        uint32_t i = static_cast<uint32_t>(clamp(frac(uv.x) * width, 0.f, width - 1.f));
        uint32_t j = static_cast<uint32_t>(clamp(frac(1.f - uv.y) * height, 0.f, height - 1.f));

        return pixels[j * width + i];
    }

    bool IsValid() const { return valid; }

private:
    bool valid{ false };
    int32_t width{};
    int32_t height{};
    uint8_t bpp{ 4 };
    vec4* pixels{};
};