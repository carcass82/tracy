/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class dielectric : public material
{
public:
    dielectric(float ri)
        : ref_idx(ri)
    {
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& s_rec) const override
    {
        s_rec.is_specular = true;
        s_rec.attenuation = vec3(1.0f, 1.0f, 1.0f);

        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);

        float ni_over_nt;
        float cosine;
        vec3 outward_normal;
        if (dot(r_in.direction(), rec.normal) > 0.0f)
        {
            outward_normal = rec.normal * -1;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / length(r_in.direction());
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = (dot(r_in.direction(), rec.normal) * -1) / length(r_in.direction());
        }

        float reflect_prob;
        vec3 refracted;
        refracted = refract(normalize(r_in.direction()), normalize(outward_normal), ni_over_nt);
        if (refracted != ZERO)
        {
            reflect_prob = schlick(cosine, ref_idx);
        }
        else
        {
            reflect_prob = 1.0f;
        }

        if (fastrand() < reflect_prob)
        {
            s_rec.specular = Ray(rec.p, reflected);
        }
        else
        {
            s_rec.specular = Ray(rec.p, refracted);
        }

        return true;
    }

private:
    const vec3 ZERO;
    float ref_idx;
};

