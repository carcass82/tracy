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

class material
{
public:
    virtual bool scatter(const Ray& r_in, const hit_record& rec, vec3& attenuation, Ray& scattered) const = 0;
    virtual vec3 emitted(const vec2& uv, const vec3& p) const { return { 0, 0, 0 }; }
    virtual bool islambertian() const { return false; }
};
