/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "material.h"
#include "random.h"
#include "collision.h"
#include "log.h"

//
// Material utility functions
//
namespace
{
    CUDA_DEVICE_CALL constexpr inline float pow2(float x)
    {
        return x * x;
    }

    CUDA_DEVICE_CALL constexpr inline float pow5(float x)
    {
        const float x2{ x * x };

        return x2 * x2 * x;
    }

    CUDA_DEVICE_CALL constexpr inline float schlick(float cos, float ref_idx)
    {
        const float r0{ pow2((1.f - ref_idx) / (1.f + ref_idx)) };

        return r0 + (1.f - r0) * pow5(1.f - cos);
    }

    CUDA_DEVICE_CALL inline vec3 random_on_unit_sphere(RandomCtx random_ctx)
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
}

CUDA_DEVICE_CALL void Material::Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const
{
    static constexpr float kRayOffset{ 0.001f };

    const vec3 raydir{ ray.GetDirection() };

    const vec3 emissive{ GetEmissive(hit) };
    const float metalness{ GetMetalness(hit) };
    const vec3 basecolor{ GetBaseColor(hit) };
    const float roughness{ GetRoughness(hit) };
    const vec3 normal{ GetNormal(hit) };
    
    const vec3 scatteredOrigin{ hit.point };
    vec3 scatteredDirection{};
    vec3 attenuation{};

    // TODO: merge BxDFs

    if (translucent_ > EPS) // BTDF
    {
        const float VdotN{ dot(raydir, normal) };
        const bool inside{ VdotN > EPS };
        const float cosine{ inside ? sqrtf(1.f - pow2(ior_) * (1.f - pow2(VdotN))) : -VdotN };
        const float ior{ inside ? ior_ : rcp(ior_) };
        
        const vec3 scattered{ normal + random_on_unit_sphere(random_ctx) };
        const vec3 reflected{ reflect(raydir, normal) };
        const vec3 specular{ lerp(reflected, scattered, roughness) };

        const vec3 refracted{ refract(raydir, normal, ior) };
        const vec3 transmitted{ lerp(refracted, scattered, roughness) };
        
        const bool is_specular{ fastrand(random_ctx) < schlick(cosine, ior) };

        scatteredDirection = is_specular ? specular : transmitted;
        attenuation = basecolor; // TODO: implement absorption
    }
    else // BRDF
    {
        const vec3 specularcolor{ lerp(vec3{ .85f }, basecolor, metalness) };

        const vec3 scattered{ normal + random_on_unit_sphere(random_ctx) };
        const vec3 reflected{ reflect(raydir, normal) };
        const vec3 specular{ lerp(reflected, scattered, roughness) };

        const float VdotN{ dot(raydir, normal) };
        
        const float materialspecularchance{ lerp(.1f, 1.f, metalness) };
        const float fresnelspecularchance{ lerp(materialspecularchance, 1.f, (1.f - roughness) * schlick(-VdotN, 1.f)) };
        
        const bool is_specular{ fastrand(random_ctx) < fresnelspecularchance };
        
        scatteredDirection = is_specular ? specular : scattered;
        attenuation = is_specular ? specularcolor : basecolor;
    }

    out_scattered = Ray(scatteredOrigin + kRayOffset * scatteredDirection, scatteredDirection);
    out_attenuation = attenuation;
    out_emission = emissive;
}

CUDA_DEVICE_CALL constexpr vec3 Material::GetBaseColor(const collision::HitData& hit) const
{
    return (HasTexture(TextureID::eBASECOLOR) ? base_color_map_.GetPixel(hit.uv).rgb : albedo_);
}

CUDA_DEVICE_CALL constexpr float Material::GetRoughness(const collision::HitData& hit) const
{
    return (HasTexture(TextureID::eROUGHNESS) ? roughness_map_.GetPixel(hit.uv).r : roughness_);
}

CUDA_DEVICE_CALL constexpr float Material::GetMetalness(const collision::HitData& hit) const
{
    return (HasTexture(TextureID::eMETALNESS) ? metalness_map_.GetPixel(hit.uv).r : metalness_);
}

CUDA_DEVICE_CALL constexpr vec3 Material::GetEmissive(const collision::HitData& hit) const
{
    return (HasTexture(TextureID::eEMISSIVE) ? emissive_map_.GetPixel(hit.uv).rgb : emissive_);
}

CUDA_DEVICE_CALL constexpr vec3 Material::GetNormal(const collision::HitData& hit) const
{
    if (HasTexture(TextureID::eNORMAL))
    {
        vec3 normal = vec3{ normal_map_.GetPixel(hit.uv).rgb } * 2.f - 1.f;
        vec3 bitangent = cross(hit.normal, normalize(hit.tangent - dot(hit.tangent, hit.normal) * hit.normal));
        mat3 tbn{ bitangent, hit.tangent, hit.normal };
        
        return normalize(tbn * normal);
    }
    else
    {
        return hit.normal;
    }
}
