#pragma once
#include "material.hpp"

class isotropic : public material
{
public:
    isotropic(texture* a)
        : albedo(a)
    {
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, vec3& attenuation, Ray& scattered) const override
    {
        scattered = Ray(rec.p, random_in_unit_sphere());
        attenuation = albedo->value(rec.uv, rec.p);
        return true;
    }

private:
    texture* albedo;
};

