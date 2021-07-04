/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once

#include "common.h"
#include "random.h"
#include "ray.h"
#include "texture.h"

class Material
{
public:
    enum TextureID { eBASECOLOR, eNORMAL, eROUGHNESS, eMETALNESS, eEMISSIVE, eCOUNT };

    Material(const vec3& in_color = vec3(), float in_roughness = 1.f, float in_metalness = .0f, float in_ior = 1.f, float in_emissive = 0.f, float in_translucency = 0.f)
        : albedo_{ in_color }
        , roughness_{ in_roughness }
        , metalness_{ in_metalness }
        , ior_{ in_ior }
        , emissive_{ in_emissive * in_color }
        , translucent_{ in_translucency }
    {}

    void SetTexture(uint32_t in_texture, TextureID in_texture_id)
    {
        switch (in_texture_id)
        {
        case TextureID::eBASECOLOR:
            base_color_map_ = in_texture;
            break;
        case TextureID::eNORMAL:
            normal_map_ = in_texture;
            break;
        case TextureID::eROUGHNESS:
            roughness_map_ = in_texture;
            break;
        case TextureID::eMETALNESS:
            metalness_map_ = in_texture;
            break;
        case TextureID::eEMISSIVE:
            emissive_map_ = in_texture;
            break;
        default:
            return;
        }

        texture_flag_[in_texture_id] = true;
    }

    uint32_t GetTexture(TextureID in_texture_id) const
    {
        if (texture_flag_[in_texture_id])
        {
            switch (in_texture_id)
            {
            case TextureID::eBASECOLOR: return base_color_map_;
            case TextureID::eNORMAL: return normal_map_;
            case TextureID::eROUGHNESS: return roughness_map_;
            case TextureID::eMETALNESS: return metalness_map_;
            case TextureID::eEMISSIVE: return emissive_map_;
            default:
                break;
            }
        }

        return UINT32_MAX;
    }

    template<class TextureProvider>
    CUDA_DEVICE void Scatter(const TextureProvider& provider, const Ray& ray, const HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const;

    template<class TextureProvider>
    CUDA_DEVICE vec3 GetBaseColor(const TextureProvider& provider, const HitData& hit) const;

    template<class TextureProvider>
    CUDA_DEVICE vec3 GetNormal(const TextureProvider& provider, const HitData& hit) const;

    template<class TextureProvider>
    CUDA_DEVICE float GetRoughness(const TextureProvider& provider, const HitData& hit) const;

    template<class TextureProvider>
    CUDA_DEVICE float GetMetalness(const TextureProvider& provider, const HitData& hit) const;

    template<class TextureProvider>
    CUDA_DEVICE vec3 GetEmissive(const TextureProvider& provider, const HitData& hit) const;

    constexpr bool HasTexture(TextureID in_texture_id) const
    {
        return texture_flag_[in_texture_id];
    }

    const vec3& GetAlbedo() const { return albedo_; }
    const vec3& GetEmissive() const { return emissive_; }
    float GetRoughness() const { return roughness_; }
    float GetMetalness() const { return metalness_; }
    float GetRefractiveIndex() const { return ior_; }
    float GetTranslucent() const { return translucent_; }

protected:
    vec3 albedo_{};
    float roughness_{ .0f };
    float metalness_{ .0f };
    float ior_{ 1.f };
    vec3 emissive_{};
    float translucent_{};

    uint32_t base_color_map_{};
    uint32_t normal_map_{};
    uint32_t roughness_map_{};
    uint32_t metalness_map_{};
    uint32_t emissive_map_{};

    bool texture_flag_[TextureID::eCOUNT]{};
};


//
// helper functions
//

static inline constexpr float pow2(float x)
{
    return x * x;
}

static inline constexpr float pow5(float x)
{
    const float x2{ x * x };

    return x2 * x2 * x;
}

static inline constexpr float schlick(float cos, float ref_idx)
{
    const float r0{ pow2((1.f - ref_idx) / (1.f + ref_idx)) };

    return r0 + (1.f - r0) * pow5(1.f - cos);
}

CUDA_DEVICE static inline CC_CONSTEXPR vec3 random_on_unit_sphere(RandomCtx random_ctx)
{
    //
    // http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    // see Method 10
    //
    const float z{ 2.f * fastrand(random_ctx) - 1.f };

    const float phi{ 2.f * PI * fastrand(random_ctx) };

    const float r{ sqrtf(1.f - pow2(z)) };

    return vec3{ r * cosf(phi), r * sinf(phi), z };
}


//
// texture helper functions
//

template<class TextureProvider>
CUDA_DEVICE inline vec3 Material::GetBaseColor(const TextureProvider& provider, const HitData& hit) const
{
    return (HasTexture(TextureID::eBASECOLOR) ? provider.GetTexture(base_color_map_).GetPixel(hit.uv).rgb : albedo_);
}

template<class TextureProvider>
CUDA_DEVICE inline float Material::GetRoughness(const TextureProvider& provider, const HitData& hit) const
{
    return (HasTexture(TextureID::eROUGHNESS) ? provider.GetTexture(roughness_map_).GetPixel(hit.uv).r : roughness_);
}

template<class TextureProvider>
CUDA_DEVICE inline float Material::GetMetalness(const TextureProvider& provider, const HitData& hit) const
{
    return (HasTexture(TextureID::eMETALNESS) ? provider.GetTexture(metalness_map_).GetPixel(hit.uv).r : metalness_);
}

template<class TextureProvider>
CUDA_DEVICE inline vec3 Material::GetEmissive(const TextureProvider& provider, const HitData& hit) const
{
    return (HasTexture(TextureID::eEMISSIVE) ? provider.GetTexture(emissive_map_).GetPixel(hit.uv).rgb : emissive_);
}

template<class TextureProvider>
CUDA_DEVICE inline vec3 Material::GetNormal(const TextureProvider& provider, const HitData& hit) const
{
    if (HasTexture(TextureID::eNORMAL))
    {
        vec3 normal = vec3{ provider.GetTexture(normal_map_).GetPixel(hit.uv).rgb } * 2.f - 1.f;
        vec3 bitangent = cross(hit.normal, normalize(hit.tangent - dot(hit.tangent, hit.normal) * hit.normal));
        mat3 tbn{ bitangent, hit.tangent, hit.normal };

        return normalize(tbn * normal);
    }
    else
    {
        return hit.normal;
    }
}


//
// actual lighting
//

template<class TextureProvider>
CUDA_DEVICE inline void Material::Scatter(const TextureProvider& provider, const Ray& ray, const HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const
{
    static constexpr float kRayOffset{ 0.001f };

    const vec3 raydir{ ray.GetDirection() };

    const vec3 emissive{ GetEmissive(provider, hit) };
    const float metalness{ GetMetalness(provider, hit) };
    const vec3 basecolor{ GetBaseColor(provider, hit) };
    const float roughness{ GetRoughness(provider, hit) };
    const vec3 normal{ GetNormal(provider, hit) };

    const vec3 scatteredOrigin{ hit.point };
    vec3 scatteredDirection{};
    vec3 attenuation{};

    // TODO: merge BxDFs

    const float VdotN{ dot(raydir, normal) };

    const vec3 scattered{ normal + random_on_unit_sphere(random_ctx) };
    const vec3 reflected{ reflect(raydir, normal) };
    const vec3 specular{ lerp(reflected, scattered, roughness) };


    if (translucent_ > EPS) // BTDF
    {
        const bool inside{ VdotN > EPS };
        const float cosine{ inside ? sqrtf(1.f - pow2(ior_) * (1.f - pow2(VdotN))) : -VdotN };
        const float ior{ inside ? ior_ : rcp(ior_) };

        const vec3 refracted{ refract(raydir, normal, ior) };
        const vec3 transmitted{ lerp(refracted, scattered, roughness) };

        const bool is_specular{ fastrand(random_ctx) < schlick(cosine, ior) };

        scatteredDirection = is_specular ? specular : transmitted;
        attenuation = basecolor; // TODO: implement absorption
    }
    else // BRDF
    {
        const vec3 specularcolor{ lerp(vec3{ .85f }, basecolor, metalness) };

        const float materialspecularchance{ lerp(.1f, 1.f, metalness) };
        const float fresnelspecularchance{ lerp(materialspecularchance, 1.f, (1.f - roughness) * schlick(-VdotN, 1.f)) };

        const bool is_specular{ fastrand(random_ctx) < fresnelspecularchance };

        scatteredDirection = is_specular ? specular : scattered;
        attenuation = is_specular ? specularcolor : basecolor;
    }

    scatteredDirection = normalize(scatteredDirection);

    out_scattered = Ray(scatteredOrigin + kRayOffset * scatteredDirection, scatteredDirection);
    out_attenuation = attenuation;
    out_emission = emissive;
}
