/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "shape.hpp"

class Triangle : public IShape
{
public:
    Triangle(vec3 a, vec3 b, vec3 c, IMaterial* ptr)
        : vertices{a, b, c}
        , mat(ptr)
    {
        normal = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        vec3 v0v1 = vertices[1] - vertices[0];
        vec3 v0v2 = vertices[2] - vertices[0];
        vec3 pvec = cross(r.get_direction(), v0v2);
        float det = dot(v0v1, pvec);

        // if the determinant is negative the triangle is backfacing
        // if the determinant is close to 0, the ray misses the triangle
        if (det < .001f)
        {
            return false;
        }

        float invDet = rcp(det);

        vec3 tvec = r.get_origin() - vertices[0];
        float u = dot(tvec, pvec) * invDet;
        if (u < .0f || u > 1.f)
        {
            return false;
        }

        vec3 qvec = cross(tvec, v0v1);
        float v = dot(r.get_direction(), qvec) * invDet;
        if (v < .0f || u + v > 1.f)
        {
            return false;
        }

        rec.t = dot(v0v2, qvec) * invDet;
        rec.uv = { u, v };
        return true;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const override final
    {
        rec.point = r.point_at(rec.t);
        rec.normal = normal;
        rec.material = mat;
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        min = min3(vertices[0], min3(vertices[1], vertices[2]));
        max = max3(vertices[0], max3(vertices[1], vertices[2]));
    }

private:
    vec3 vertices[3];
    vec3 normal;
    IMaterial* mat;
};
