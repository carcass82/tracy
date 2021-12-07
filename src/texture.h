/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#include "common.h"

class Texture
{
public:
    Texture()
    {}

    template<typename T>
    Texture(i32 in_width, i32 in_height, T* in_pixels, bool sRGB = false);

    ~Texture()
    {
        delete[] pixels_;
    }

    // disable copying
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept
        : valid_{ other.valid_ }
        , width_{ other.width_ }
        , height_{ other.height_ }
        , bpp_{ other.bpp_ }
        , pixels_{ std::exchange(other.pixels_, nullptr) }
    {}

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            valid_ = std::move(other.valid_);
            width_ = std::move(other.width_);
            height_ = std::move(other.height_);
            bpp_ = std::move(other.bpp_);
            pixels_ = std::exchange(other.pixels_, nullptr);
        }
        return *this;
    }

    CUDA_DEVICE const vec4& GetPixel(const vec2& uv) const
    {
        // TODO: assuming "GL_REPEAT" mode, implement other behaviors
        u32 i = static_cast<u32>(clamp(frac(uv.x) * width_, 0.f, width_ - 1.f));
        u32 j = static_cast<u32>(clamp(frac(1.f - uv.y) * height_, 0.f, height_ - 1.f));

        return pixels_[j * width_ + i];
    }

    constexpr u32 GetWidth() const
    {
        return width_;
    }

    constexpr u32 GetHeight() const
    {
        return height_;
    }

    constexpr u32 GetBPP() const
    {
        return 4;
    }

    constexpr vec4* GetPixels() const
    {
        return pixels_;
    }

    constexpr bool IsValid() const
    {
        return valid_;
    }


private:
    bool valid_{ false };
    i32 width_{};
    i32 height_{};
    u8 bpp_{ 4 };
    vec4* pixels_{};
};

template<typename T>
inline Texture::Texture(i32 in_width, i32 in_height, T* in_pixels, bool sRGB)
    : valid_{ true }, width_{ in_width }, height_{ in_height }, pixels_{ new vec4[in_width * in_height] }
{
    // only handle unsigned char and float types
    static_assert(std::is_same<T, u8>::value || std::is_same<T, float>::value, "Unknown pixel data format");

    // floating point texture are expected to have a value of 0...1, no need to remap from 0...255 range
    static constexpr float kRemap = std::is_same<T, float>::value ? 1.f : 255.f;

    for (i32 i = 0; i < width_ * height_; ++i)
    {
        vec4 pixel = vec4(in_pixels[i * bpp_], in_pixels[i * bpp_ + 1], in_pixels[i * bpp_ + 2], in_pixels[i * bpp_ + 3]) / kRemap;
        pixels_[i] = sRGB ? linear(pixel) : pixel;
    }
}

template<>
inline Texture::Texture(i32 in_width, i32 in_height, vec4* in_pixels, bool sRGB)
    : valid_{ true }, width_{ in_width }, height_{ in_height }, pixels_{ in_pixels }
{
}
