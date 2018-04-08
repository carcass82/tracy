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

    virtual bool scatter(const Ray& r_in, const hit_record& rec, vec3& attenuation, Ray& scattered) const override
    {
        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + random_in_unit_sphere() * fuzz);
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

private:
    vec3 albedo;
    float fuzz;
};

