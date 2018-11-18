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
    Triangle(vec3 a, vec3 b, vec3 c, IMaterial* m)
        : vert{a, b, c}
        , mat(m)
    {
        normal = normalize(cross(vert[1] - vert[0], vert[2] - vert[0]));
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        vec3 v0v1 = vert[1] - vert[0];
        vec3 v0v2 = vert[2] - vert[0];
        vec3 pvec = cross(r.get_direction(), v0v2);
        float det = dot(v0v1, pvec);

        // if the determinant is negative the triangle is backfacing
        // if the determinant is close to 0, the ray misses the triangle
        if (det < .001f)
        {
            return false;
        }

        float invDet = rcp(det);

        vec3 tvec = r.get_origin() - vert[0];
        float u = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1)
        {
            return false;
        }

        vec3 qvec = cross(tvec, v0v1);
        float v = dot(r.get_direction(), qvec) * invDet;
        if (v < 0 || u + v > 1)
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

private:
    vec3 vert[3];
    vec3 normal;
    IMaterial* mat;
};
