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
    Box() {}

    Box(const vec3& p0, const vec3& p1, IMaterial* ptr)
        : pmin(p0)
        , pmax(p1)
        , mat(ptr)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override final
    {
        float tmin = t_min;
        float tmax = t_max;

        for (int i = 0; i < 3; ++i)
        {
            float direction = r.get_direction()[i];
            float origin = r.get_origin()[i];
            float minbound = pmin[i];
            float maxbound = pmax[i];

            float ood = rcp(direction);
            float t1 = (minbound - origin) * ood;
            float t2 = (maxbound - origin) * ood;

            if (t1 > t2) swap(t1, t2);

            tmin = max(tmin, t1);
            tmax = min(tmax, t2);

            if (tmin > tmax || tmin > t_max)
            {
                return false;
            }
        }

        rec.t = tmin;
        return true;
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

    void expand(const vec3& v)
    {
        pmin = min3(pmin, v);
        pmax = max3(pmax, v);
    }

    bool contains(IShape* object)
    {
        bool contained(false);

        if (object)
        {
            vec3 bmin, bmax;
            object->get_bounds(bmin, bmax);

            contained = (bmin.x >= pmin.x && bmin.x <= pmax.x) ||
                        (bmin.y >= pmin.y && bmin.y <= pmax.y) ||
                        (bmin.z >= pmin.z && bmin.z <= pmax.z) ||
                        (bmax.x >= pmin.x && bmax.x <= pmax.x) ||
                        (bmax.y >= pmin.y && bmax.y <= pmax.y) ||
                        (bmax.z >= pmin.z && bmax.z <= pmax.z);
        }

        return contained;
    }

private:
    vec3 get_normal(const vec3& point) const
    {
        const float eps = .001f;

        if (fabsf(pmin.x - point.x) < eps) return vec3{ -1.0f,   .0f,   .0f };
        if (fabsf(pmax.x - point.x) < eps) return vec3{  1.0f,   .0f,   .0f };
        if (fabsf(pmin.y - point.y) < eps) return vec3{   .0f, -1.0f,   .0f };
        if (fabsf(pmax.y - point.y) < eps) return vec3{   .0f,  1.0f,   .0f };
        if (fabsf(pmin.z - point.z) < eps) return vec3{   .0f,   .0f, -1.0f };
        return vec3{ .0f, .0f, 1.0f };
    }

    vec2 get_uv(const vec3& point) const
    {
        const float eps = .001f;

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
