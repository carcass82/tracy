/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once
#include "geom.hpp"
#include "texture.hpp"
#include "ray.hpp"

using vmath::vec3;
using vmath::normalize;
using vmath::dot;
using vutil::max;

class material;

struct hit_record
{
    float t;
    float u;
    float v;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class material
{
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
    virtual vec3 emitted(float u, float v, const vec3& p) const { return vec3(); }
};

class lambertian : public material
{
public:
    lambertian(texture* a)
        : albedo(a)
    {
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const override
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo->value(rec.u, rec.v, rec.p);

        return true;
    }

private:
    texture* albedo;
};

class metal : public material
{
public:
    metal(const vec3& a, float f)
        : albedo(a)
    {
        fuzz = max(f, 1.0f);
    }
    
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const override
    {
        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + random_in_unit_sphere() * fuzz);
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

private:
    vec3 albedo;
    float fuzz;
};

class dielectric : public material
{
public:
    dielectric(float ri)
        : ref_idx(ri)
    {
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const override
    {
        vec3 outward_normal;
        vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);

        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;

        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = rec.normal * -1;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / length(r_in.direction());
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = (dot(r_in.direction(), rec.normal) * -1) / length(r_in.direction());
        }

		const static vec3 ZERO;
        refracted = refract(normalize(r_in.direction()), normalize(outward_normal), ni_over_nt);
		if (refracted != ZERO) {
            reflect_prob = schlick(cosine, ref_idx);
        } else {
            reflect_prob = 1.0f;
        }

        if (fastrand() < reflect_prob) {
            scattered = ray(rec.p, reflected);
        } else {
            scattered = ray(rec.p, refracted);
        }

        return true;
    }

private:
    float ref_idx;
};


class diffuse_light : public material
{
public:
    diffuse_light(texture* a)
        : emit(a)
    {
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const override
    {
        return false;
    }

    virtual vec3 emitted(float u, float v, const vec3& p) const override
    {
        return emit->value(u, v, p);
    }

private:
    texture* emit;
};


class isotropic : public material
{
public:
    isotropic(texture* a)
        : albedo(a)
    {
    }

    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const override	
    {
        scattered = ray(rec.p, random_in_unit_sphere());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

private:
    texture* albedo;
};
