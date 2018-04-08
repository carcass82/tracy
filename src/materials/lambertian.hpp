#pragma once
#include "material.hpp"

class lambertian : public material
{
public:
    lambertian(texture* a)
        : albedo(a)
    {
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, vec3& attenuation, Ray& scattered) const override
    {
        //
        // random_in_unit_sphere is not good for lambertian materials
        // see: http://aras-p.info/blog/2018/03/31/Daily-Pathtracer-Part-4-Fixes--Mitsuba/
        //
        vec3 target = rec.p + rec.normal + random_unit_vector();
        scattered = Ray(rec.p, target - rec.p);
        attenuation = albedo->value(rec.uv, rec.p);

        return true;
    }

    virtual bool islambertian() const override { return true; }

private:
    texture* albedo;
};

