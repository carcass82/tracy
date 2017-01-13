/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/epsilon.hpp"

#include "geom.hpp"
#include "texture.hpp"
#include "ray.hpp"

class material;

struct hit_record
{
    float t;
    float u;
    float v;
    glm::vec3 p;
    glm::vec3 normal;
    material* mat_ptr;
};

class material
{
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const = 0;
    virtual glm::vec3 emitted(float u, float v, const glm::vec3& p) const { return glm::vec3(); }
};

class lambertian : public material
{
public:
    lambertian(texture* a) : albedo(a) {}

    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const override
    {
        glm::vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo->value(rec.u, rec.v, rec.p);

        return true;
    }

    texture* albedo;
};

class metal : public material
{
public:
    metal(const glm::vec3& a, float f) : albedo(a) { fuzz = (f < 1.0f)? f : 1.0f; }
    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const override
    {
        glm::vec3 reflected = reflect(glm::normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

    glm::vec3 albedo;
    float fuzz;
};

class dielectric : public material
{
public:
    dielectric(float ri) : ref_idx(ri) {}

    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const override
    {
        glm::vec3 outward_normal;
        glm::vec3 reflected = reflect(glm::normalize(r_in.direction()), rec.normal);

        float ni_over_nt;
        attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 refracted;

        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        refracted = refract(normalize(r_in.direction()), normalize(outward_normal), ni_over_nt);
        if (!(glm::epsilonEqual(refracted[0], 0.0f, FLT_EPSILON) && glm::epsilonEqual(refracted[1], 0.0f, FLT_EPSILON) && glm::epsilonEqual(refracted[1], 0.0f, FLT_EPSILON))) {
            reflect_prob = schlick(cosine, ref_idx);
        } else {
            reflect_prob = 1.0f;
        }

        if (drand48() < reflect_prob) {
            scattered = ray(rec.p, reflected);
        } else {
            scattered = ray(rec.p, refracted);
        }

        return true;
    }

    float ref_idx;
};


class diffuse_light : public material
{
public:
    diffuse_light(texture* a) : emit(a) {}

    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const override
    {
        return false;
    }

    virtual glm::vec3 emitted(float u, float v, const glm::vec3& p) const override
    {
        return emit->value(u, v, p);
    }

    texture* emit;
};


class isotropic : public material
{
public:
    isotropic(texture* a) : albedo(a) {}

    virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered) const override	
    {
        scattered = ray(rec.p, random_in_unit_sphere());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    texture* albedo;
};
