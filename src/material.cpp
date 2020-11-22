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

CUDA_DEVICE_CALL bool Material::Scatter(const Ray& ray, const collision::HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered, RandomCtx random_ctx) const
{
    static constexpr float kRayOffset{ 0.001f };

    vec3 attenuation{};
    vec3 emission{};
    vec3 scatteredDirection{};
    vec3 scatteredOrigin{};
    bool scattered{ false };

    switch (material_type_)
    {
    case MaterialID::eLAMBERTIAN:
    {
        vec3 target{ hit.point + hit.normal + random_on_unit_sphere(random_ctx) };
        scatteredDirection = normalize(target - hit.point);
        scatteredOrigin = hit.point;
        attenuation = albedo_;
        scattered = true;
        break;
    }

    case MaterialID::eMETAL:
    {
        vec3 reflected = reflect(ray.GetDirection(), hit.normal);
        scatteredOrigin = hit.point;
        scatteredDirection = reflected + roughness_ * random_on_unit_sphere(random_ctx);
        attenuation = albedo_;
        scattered = dot(scatteredDirection, hit.normal) > .0f;
        break;
    }

    case MaterialID::eDIELECTRIC:
    {
        vec3 outward_normal{ hit.normal };
        float ni_nt;
        float cosine = dot(ray.GetDirection(), hit.normal);

        if (cosine > EPS)
        {
            outward_normal *= -1.f;
            ni_nt = ior_;
            cosine = sqrtf(1.f - ior_ * ior_ * (1.f - cosine - cosine));
        }
        else
        {
            ni_nt = rcp(ior_);
            cosine *= -1;
        }

        vec3 refracted = refract(ray.GetDirection(), outward_normal, ni_nt);
        vec3 reflected = reflect(ray.GetDirection(), hit.normal);
        float reflect_chance = (refracted != vec3{}) ? schlick(cosine, ior_) : 1.0f;

        scatteredOrigin = hit.point;
        scatteredDirection = (fastrand(random_ctx) < reflect_chance)? reflected : refracted;
        attenuation = albedo_;
        scattered = true;
        break;
    }

    case MaterialID::eEMISSIVE:
    {
        emission = albedo_;
        scattered = false;
        break;
    }
    }

    out_scattered = Ray(scatteredOrigin + kRayOffset * scatteredDirection, scatteredDirection);
    out_attenuation = attenuation;
    out_emission = emission;
    return scattered;
}
