/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "material.hpp"

class lambertian : public material
{
public:
    lambertian(Texture* a)
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
        float cosine = max(.0f, dot(rec.normal, normalize(scattered.GetDirection())));
        return cosine / PI;
    }

    virtual bool islambertian() const override { return true; }

private:
    Texture* albedo;
};

