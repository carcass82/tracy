/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "common.h"
#include "ray.h"

namespace collision { struct HitData; }

struct Texture
{
    Texture()
    {}

    Texture(int32_t in_width, int32_t in_height, uint8_t* in_pixels, bool sRGB = false)
        : width{ in_width }, height{ in_height }
    {
        pixels = new vec4[width * height];

        for (int32_t i = 0; i < width * height; ++i)
        {
            vec4 pixel = vec4(in_pixels[i * bpp], in_pixels[i * bpp + 1], in_pixels[i * bpp + 2], in_pixels[i * bpp + 3]) / 255.f;
            pixels[i] = sRGB? linear(pixel) : pixel;
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
        : width{ other.width }, height{ other.height }, bpp{ other.bpp }, pixels{ std::exchange(other.pixels, nullptr) }
    {}

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            width = std::move(other.width);
            height = std::move(other.height);
            bpp = std::move(other.bpp);
            pixels = std::exchange(other.pixels, nullptr);
        }
        return *this;
    }

    vec4 GetPixel(const vec2& uv) const
    {
        uint32_t i = static_cast<uint32_t>(clamp(frac(uv.x) * width, 0.f, width - 1.f));
        uint32_t j = static_cast<uint32_t>(clamp(frac(1.f - uv.y) * height, 0.f, height - 1.f));

        return pixels[j * width + i];
    }

    float frac(float x) const { return x - trunc(x); }
    
    int32_t width{};
    int32_t height{};
    uint8_t bpp{ 4 };
    vec4* pixels{};
};

class Material
{
public:
    enum class MaterialID { eINVALID, eLAMBERTIAN, eMETAL, eDIELECTRIC, eEMISSIVE };
    enum class TextureID { eBASECOLOR, eNORMAL, eROUGHNESS, eMETALNESS, eEMISSIVE };

    CUDA_DEVICE_CALL Material()
    {}

    CUDA_DEVICE_CALL Material(MaterialID in_type, const vec3& in_color, float in_roughness = .0f, float in_ior = 1.f)
        : material_type_{ in_type }
        , albedo_{ in_color }
        , roughness_{ in_roughness }
        , ior_{ in_ior }
        , emissive_{ (in_type == MaterialID::eEMISSIVE)? in_color : vec3{} }
    {}

    void SetTexture(Texture&& in_texture, TextureID in_texture_id)
    {
        switch (in_texture_id)
        {
        case TextureID::eBASECOLOR:
            base_color_map_ = std::move(in_texture);
            break;
        case TextureID::eNORMAL:
            normal_map_ = std::move(in_texture);
            break;
        case TextureID::eROUGHNESS:
            roughness_map_ = std::move(in_texture);
            break;
        case TextureID::eMETALNESS:
            metalness_map_ = std::move(in_texture);
            break;
        case TextureID::eEMISSIVE:
            emissive_map_ = std::move(in_texture);
            break;
        }
    }

    CUDA_DEVICE_CALL bool Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const;

    vec3 GetBaseColor(const collision::HitData& hit) const;

    vec3 GetNormal(const collision::HitData& hit) const;

    float GetRoughness(const collision::HitData& hit) const;

    float GetMetalness(const collision::HitData& hit) const;

    vec3 GetEmissive(const collision::HitData& hit) const;

private:

    MaterialID material_type_{ MaterialID::eINVALID };
    vec3 albedo_{};
    float roughness_{ .0f };
    float ior_{ 1.f };
    vec3 emissive_{};

    Texture base_color_map_{};
    Texture normal_map_{};
    Texture roughness_map_{};
    Texture metalness_map_{};
    Texture emissive_map_{};
};
