/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "geom.hpp"
#include "textures/texture.hpp"
#include "ray.hpp"

class IMaterial;

struct HitData
{
    float t;
    vec2 uv;
    vec3 point;
    vec3 normal;
    IMaterial* material;
};

struct ScatterData
{
    Ray scattered;
    vec3 attenuation;
};


class NOVTABLE IMaterial
{
public:
    virtual bool scatter(const Ray& r_in, const HitData& rec, ScatterData& s_rec) const = 0;
    virtual vec3 emitted(const Ray& r_in, const HitData& rec, const vec2& uv, const vec3& p) const = 0;
};
