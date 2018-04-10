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
using vmath::vec2;
using vmath::normalize;
using vmath::dot;
using vutil::min;

class material;
struct hit_record
{
    float t;
    vec2 uv;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class pdf;
struct scatter_record
{
    Ray specular;
    bool is_specular;
    vec3 attenuation;
};

class material
{
public:
    virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& s_rec) const { return false; }
    virtual float scattering_pdf(const Ray& r_in, const hit_record& rec, const Ray& scattered) const { return .0f; }
    virtual vec3 emitted(const Ray& r_in, const hit_record& rec, const vec2& uv, const vec3& p) const { return vec3{ 0, 0, 0 }; }
    virtual bool islambertian() const { return false; }
};
