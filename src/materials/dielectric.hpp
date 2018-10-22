/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class Dielectric : public IMaterial
{
public:
    Dielectric(float ri)
        : ref_idx(ri)
    {
    }

    bool scatter(const Ray& r_in, const HitData& rec, ScatterData& s_rec) const override final
    {
        s_rec.attenuation = vec3(1.0f, 1.0f, 1.0f);

        vec3 reflected = reflect(normalize(r_in.GetDirection()), rec.normal);

        float ni_over_nt;
        float cosine;
        vec3 outward_normal;
        if (dot(r_in.GetDirection(), rec.normal) > 0.0f)
        {
            outward_normal = rec.normal * -1;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.GetDirection(), rec.normal) / length(r_in.GetDirection());
			cosine = sqrtf(1.f - ref_idx * ref_idx * (1.f - cosine - cosine));
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.GetDirection(), rec.normal) / length(r_in.GetDirection());
        }

        vec3 refracted = refract(normalize(r_in.GetDirection()), normalize(outward_normal), ni_over_nt);
        float reflect_prob = (refracted != ZERO)? schlick(cosine, ref_idx) : 1.f;

        s_rec.scattered = (fastrand() < reflect_prob)? Ray(rec.p, reflected) : Ray(rec.p, refracted);

        return true;
    }

    vec3 emitted(const Ray& r_in, const HitData& rec, const vec2& uv, const vec3& p) const override final
    {
        return ZERO;
    }

private:
    const vec3 ZERO;
    float ref_idx;
};

