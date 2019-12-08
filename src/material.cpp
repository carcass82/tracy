/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#include "material.h"
#include "random.h"

//
// Material utility functions
//
namespace
{
    vec3 random_on_unit_sphere()
    {
        float z = fastrand() * 2.f - 1.f;
        float a = fastrand() * 2.f * PI;
        float r = sqrtf(max(.0f, 1.f - z * z));

        return vec3{ r * cc::math::fast::cosf(a), r * cc::math::fast::sinf(a), z };
    }

    float schlick(float cos, float ref_idx)
    {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 *= r0;
        return r0 + (1.0f - r0) * powf((1.f - cos), 5.f);
    }
}

bool Material::Scatter(const Ray& ray, const HitData& hit, vec3& out_attenuation, vec3& out_emission, Ray& out_scattered) const
{
    switch (material_type_)
    {
    case MaterialID::eLAMBERTIAN:
    {
        vec3 target = hit.point + hit.normal + random_on_unit_sphere();
        out_scattered = Ray(hit.point, normalize(target - hit.point));
        out_attenuation = albedo_;
        out_emission = vec3{};

        return true;
    }

    case MaterialID::eMETAL:
    {
        vec3 reflected = reflect(ray.GetDirection(), hit.normal);
        out_scattered = Ray(hit.point, reflected + roughness_ * random_on_unit_sphere());
        out_attenuation = albedo_;
        out_emission = vec3{};

        return (dot(out_scattered.GetDirection(), hit.normal) > .0f);
    }

    case MaterialID::eDIELECTRIC:
    {
        out_attenuation = vec3{ 1.f, 1.f, 1.f };
        out_emission = vec3{};

        vec3 outward_normal;
        float ni_nt;
        float cosine;
        if (dot(ray.GetDirection(), hit.normal) > .0f)
        {
            outward_normal = -1.f * hit.normal;
            ni_nt = ior_;
            cosine = dot(ray.GetDirection(), hit.normal);
            cosine = sqrtf(1.f - ior_ * ior_ * (1.f - cosine - cosine));
        }
        else
        {
            outward_normal = hit.normal;
            ni_nt = 1.f / ior_;
            cosine = -dot(ray.GetDirection(), hit.normal);
        }

        const vec3 ZERO{};
        vec3 refracted = refract(ray.GetDirection(), outward_normal, ni_nt);
        vec3 reflected = reflect(ray.GetDirection(), hit.normal);
        float reflect_chance = (refracted != ZERO) ? schlick(cosine, ior_) : 1.0f;

        out_scattered = Ray(hit.point, (fastrand() < reflect_chance) ? reflected : refracted);
        return true;
    }

    case MaterialID::eEMISSIVE:
    {
        out_emission = albedo_;
        return false;
    }

    default:
        return false;
    }
}
