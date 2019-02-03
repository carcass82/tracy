/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "shape.hpp"

class Box : public IShape
{
public:
    Box()
    {
    }

    Box(const vec3& p0, const vec3& p1, IMaterial* ptr)
        : pmin(p0)
        , pmax(p1)
        , mat(ptr)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        // see shapelist
        return false;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const override final
    {
        rec.point = r.point_at(rec.t);
        rec.normal = get_normal(rec.point);
        rec.uv = get_uv(rec.point);
        rec.material = mat;
    }

    virtual void get_bounds(vec3& min, vec3& max) const override final
    {
        min = pmin;
        max = pmax;
    }

    virtual IMaterial* get_material() const override final
    {
        return mat;
    }

    void expand(const vec3& v)
    {
        pmin = min3(pmin, v);
        pmax = max3(pmax, v);
    }

    virtual uint32_t get_id() const override final
    {
        return make_id('B', 'O', 'X');
    }

    friend class ShapeList;

private:
    vec3 get_normal(const vec3& point) const
    {
        const float eps = 1e-6f;

        if (fabsf(pmin.x - point.x) < eps) return vec3{ -1.0f,   .0f,   .0f };
        if (fabsf(pmax.x - point.x) < eps) return vec3{  1.0f,   .0f,   .0f };
        if (fabsf(pmin.y - point.y) < eps) return vec3{   .0f, -1.0f,   .0f };
        if (fabsf(pmax.y - point.y) < eps) return vec3{   .0f,  1.0f,   .0f };
        if (fabsf(pmin.z - point.z) < eps) return vec3{   .0f,   .0f, -1.0f };
        return vec3{ .0f, .0f, 1.0f };
    }

    vec2 get_uv(const vec3& point) const
    {
        const float eps = 1e-6f;

        if ((fabsf(pmin.x - point.x) < eps) || (fabsf(pmax.x - point.x) < eps))
        {
            return vec2((point.y - pmin.y) / (pmax.y - pmin.y), (point.z - pmin.z) / (pmax.z - pmin.z));
        }
        if ((fabsf(pmin.y - point.y) < eps) || (fabsf(pmax.y - point.y) < eps))
        {
            return vec2((point.x - pmin.x) / (pmax.x - pmin.x), (point.z - pmin.z) / (pmax.z - pmin.z));
        }
        return vec2((point.x - pmin.x) / (pmax.x - pmin.x), (point.y - pmin.y) / (pmax.y - pmin.y));
    }

    vec3 pmin;
    vec3 pmax;
    IMaterial* mat;
};
