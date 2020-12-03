/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "material.h"
#include "random.h"
#include "collision.h"

//
// Material utility functions
//
namespace
{
    CUDA_DEVICE_CALL inline vec3 random_on_unit_sphere(RandomCtx random_ctx)
    {
        float z = 1.f - fastrand(random_ctx) * 2.f;
        float r = sqrtf(1.f - z * z);

        float a = fastrand(random_ctx) * 2.f * PI;

        float sin_a, cos_a;
        sincosf(a, &sin_a, &cos_a);

        return vec3(r * cos_a, r * sin_a, z);
    }

    CUDA_DEVICE_CALL constexpr inline float pow2(float x)
    {
        return x * x;
    }

    CUDA_DEVICE_CALL constexpr inline float pow5(float x)
    {
        float x2 = pow2(x);
        return x2 * x2 * x;
    }

    CUDA_DEVICE_CALL constexpr inline float schlick(float cos, float ref_idx)
    {
        float r0 = pow2((1.0f - ref_idx) / (1.0f + ref_idx));
        return r0 + (1.0f - r0) * pow5(1.f - cos);
    }
}

CUDA_DEVICE_CALL void Material::Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const
{
    static constexpr float kRayOffset{ 0.001f };

    vec3 emissive{ GetEmissive(hit) };
    float metalness{ GetMetalness(hit) };
    vec3 basecolor{ GetBaseColor(hit) };
    float roughness{ GetRoughness(hit) };
    
    // (1-metalness) * baseColor + metalness * (mix(0.04, baseColor, metalness))
    vec3 attenuation{ lerp(basecolor, lerp(vec3{ .04f }, basecolor, metalness), metalness) };
    vec3 refraction{ vec3{ 1.f } - attenuation };

    vec3 normal{ GetNormal(hit) };

    vec3 scatteredDirection{};
    vec3 scatteredOrigin{ hit.point };
    
    // translucent here is loosely mapped to Disney BxDF "specTrans" parameter
    //
    // 2 step mix:
    // translucent blends between full dielectric and full specular
    // then metalness blends the result with full metal

    // dielectric
    vec3 scattered{ normal + random_on_unit_sphere(random_ctx) };

    // specular
    float VdotN{ dot(ray.GetDirection(), normal) };
    bool inside{ VdotN > EPS };
    float cosine{ inside ? sqrtf(1.f - pow2(ior_) * (1.f - pow2(VdotN))) : -VdotN };
    vec3 reflected{ reflect(ray.GetDirection(), normal) };
    vec3 refracted{ refract(ray.GetDirection(), inside ? -normal : normal, inside ? ior_ : rcp(ior_)) };
    vec3 specular{ fastrand(random_ctx) < schlick(cosine, ior_)? reflected : refracted };

    // metallic
    vec3 metallic{ lerp(reflected, scattered, roughness) };

    // final mix
    scatteredDirection = lerp(lerp(scattered, specular, translucent_), metallic, metalness);
    
    out_scattered = Ray(scatteredOrigin + kRayOffset * scatteredDirection, scatteredDirection);
    out_attenuation = attenuation;
    out_emission = emissive;
}

vec3 Material::GetBaseColor(const collision::HitData& hit) const
{
    vec3 result = albedo_;

    if (*base_color_map_)
    {
       result = base_color_map_.GetPixel(hit.uv).rgb;
    }

    return result;
}

vec3 Material::GetNormal(const collision::HitData& hit) const
{
    if (*normal_map_)
    {
        vec3 normal = vec3{ normal_map_.GetPixel(hit.uv).rgb } * 2.f - 1.f;
        vec3 bitangent = cross(hit.normal, normalize(hit.tangent - dot(hit.tangent, hit.normal) * hit.normal));
        mat3 tbn{ bitangent, hit.tangent, hit.normal };
        
        return normalize(tbn * normal);
    }

    return hit.normal;
}

float Material::GetRoughness(const collision::HitData& hit) const
{
    float result = roughness_;

    if (*roughness_map_)
    {
        result = roughness_map_.GetPixel(hit.uv).r;
    }

    return result;
}

float Material::GetMetalness(const collision::HitData& hit) const
{
    if (*metalness_map_)
    {
        return metalness_map_.GetPixel(hit.uv).r;
    }

    return metalness_;
}

vec3 Material::GetEmissive(const collision::HitData& hit) const
{
    vec3 result = emissive_;

    if (*emissive_map_)
    {
        result = emissive_map_.GetPixel(hit.uv).rgb;
    }

    return result;
}
