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
    CUDA_DEVICE_CALL Texture()
    {}

    template<typename T>
    CUDA_DEVICE_CALL Texture(int32_t in_width, int32_t in_height, T* in_pixels, bool sRGB = false)
        : width{ in_width }, height{ in_height }, pixels{ new vec4[in_width * in_height] }, valid{ true }
    {
        // only handle unsigned char and float types
        static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, float>);

        // floating point texture are expected to have a value of 0...1, no need to remap from 0...255 range
        static constexpr float kRemap = std::is_same_v<T, float>? 1.f : 255.f;

        for (int32_t i = 0; i < width * height; ++i)
        {
            vec4 pixel = vec4(in_pixels[i * bpp], in_pixels[i * bpp + 1], in_pixels[i * bpp + 2], in_pixels[i * bpp + 3]) / kRemap;
            pixels[i] = sRGB ? linear(pixel) : pixel;
        }
    }

    CUDA_DEVICE_CALL ~Texture()
    {
        delete[] pixels;
    }

    // disable copying
    CUDA_DEVICE_CALL Texture(const Texture&) = delete;
    CUDA_DEVICE_CALL Texture& operator=(const Texture&) = delete;

    CUDA_DEVICE_CALL Texture(Texture&& other) noexcept
        : width{ other.width }, height{ other.height }, bpp{ other.bpp }, pixels{ std::exchange(other.pixels, nullptr) }, valid{ other.valid }
    {}

    CUDA_DEVICE_CALL Texture& operator=(Texture&& other) noexcept
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

    CUDA_DEVICE_CALL const vec4& GetPixel(const vec2& uv) const
    {
        // TODO: assuming "GL_REPEAT" mode, implement other behaviors
        uint32_t i = static_cast<uint32_t>(clamp(frac(uv.x) * width, 0.f, width - 1.f));
        uint32_t j = static_cast<uint32_t>(clamp(frac(1.f - uv.y) * height, 0.f, height - 1.f));

        return pixels[j * width + i];
    }

    CUDA_DEVICE_CALL vec4* Pixels()
    {
        return pixels;
    }

    CUDA_DEVICE_CALL bool IsValid() const
    {
        return valid;
    }


private:
    bool valid{ false };
    int32_t width{};
    int32_t height{};
    uint8_t bpp{ 4 };
    vec4* pixels{};
};