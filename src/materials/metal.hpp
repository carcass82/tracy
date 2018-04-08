#pragma once
#include "material.hpp"

class metal : public material
{
public:
    metal(const vec3& a, float f)
        : albedo(a)
    {
        fuzz = min(f, 1.0f);
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& s_rec) const override
    {
        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);

        s_rec.is_specular = true;
        s_rec.specular = Ray(rec.p, reflected + random_in_unit_sphere() * fuzz);
        s_rec.attenuation = albedo;

        return true;
    }

private:
    vec3 albedo;
    float fuzz;
};

