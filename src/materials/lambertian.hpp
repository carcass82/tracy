#pragma once
#include "material.hpp"

class lambertian : public material
{
public:
    lambertian(texture* a)
        : albedo(a)
    {
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& srec) const override
    {
        srec.is_specular = false;
        srec.attenuation = albedo->value(rec.uv, rec.p);

        return true;
    }

    virtual float scattering_pdf(const Ray &r_in, const hit_record &rec, const Ray &scattered) const
    {
        float cosine = max(.0f, dot(rec.normal, normalize(scattered.direction())));
        return cosine / PI;
    }

    virtual bool islambertian() const override { return true; }

private:
    texture* albedo;
};

