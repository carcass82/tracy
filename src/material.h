/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "common.h"
#include "ray.h"
#include "texture.h"

namespace collision { struct HitData; }

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
        , metalness_{ in_type == MaterialID::eMETAL? 1.f : 0.f }
        , ior_{ in_ior }
        , emissive_{ in_type == MaterialID::eEMISSIVE? in_color : vec3{} }
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

    CUDA_DEVICE_CALL void Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const;

    vec3 GetBaseColor(const collision::HitData& hit) const;

    vec3 GetNormal(const collision::HitData& hit) const;

    float GetRoughness(const collision::HitData& hit) const;

    float GetMetalness(const collision::HitData& hit) const;

    vec3 GetEmissive(const collision::HitData& hit) const;

private:

    MaterialID material_type_{ MaterialID::eINVALID };
    vec3 albedo_{};
    float roughness_{ .0f };
    float metalness_{ .0f };
    float ior_{ 1.f };
    vec3 emissive_{};

    Texture base_color_map_{};
    Texture normal_map_{};
    Texture roughness_map_{};
    Texture metalness_map_{};
    Texture emissive_map_{};
};
