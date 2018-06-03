/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "geom.hpp"
#include "texture.hpp"
#include "ray.hpp"

using cc::math::vec3;
using cc::math::vec2;
using cc::math::normalize;
using cc::math::dot;
using cc::util::min;

class material;
class pdf;

struct hit_record
{
    float t;
    vec2 uv;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

struct scatter_record
{
    Ray specular;
    bool is_specular;
    vec3 attenuation;
};


class material
{
public:
    virtual bool scatter(const Ray& r_in, const hit_record& rec, scatter_record& s_rec) const          { return false; }
    virtual float scattering_pdf(const Ray& r_in, const hit_record& rec, const Ray& scattered) const   { return .0f; }
    virtual vec3 emitted(const Ray& r_in, const hit_record& rec, const vec2& uv, const vec3& p) const  { return vec3(); }
    virtual bool islambertian() const                                                                  { return false; }
};
