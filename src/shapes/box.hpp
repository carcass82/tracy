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

    virtual bool hit(const Ray& r, float t_min, float t_max, HitData& rec) const override
    {
        float tmin = t_min;
        float tmax = FLT_MAX;

        for (int i = 0; i < 3; ++i)
        {
            float direction = r.GetDirection()[i];
            float origin = r.GetOrigin()[i];
            float minbound = pmin[i];
            float maxbound = pmax[i];

            if (fabsf(direction) < .0001f)
            {
                if (origin < minbound || origin > maxbound) return false;
            }
            else
            {
                float ood = cc::math::fast::rcp(direction);
                float t1 = (minbound - origin) * ood;
                float t2 = (maxbound - origin) * ood;

                if (t1 > t2) cc::util::swap(t1, t2);

                tmin = max(tmin, t1);
                tmax = min(tmax, t2);

                if (tmin > tmax || tmin > t_max) return false;
            }
        }

        rec.t = tmin;
        return true;
    }

    virtual void get_hit_data(const Ray& r, HitData& rec) const
    {
        rec.p = r.PointAt(rec.t);
        rec.normal = get_normal(rec.p);
        rec.uv = get_uv(rec.p);
        rec.mat_ptr = mat;
    }

private:
    vec3 get_normal(const vec3& point) const
    {
        if (fabsf(pmin.x - point.x) < cc::math::EPS) return vec3(-1.f, .0f, .0f);
        if (fabsf(pmax.x - point.x) < cc::math::EPS) return vec3(1.f, .0f, .0f);
        if (fabsf(pmin.y - point.y) < cc::math::EPS) return vec3(.0f, -1.f, .0f);
        if (fabsf(pmax.y - point.y) < cc::math::EPS) return vec3(.0f, 1.f, .0f);
        if (fabsf(pmin.z - point.z) < cc::math::EPS) return vec3(.0f, .0f, -1.f);
        return vec3(.0f, .0f, 1.f);
    }

    vec2 get_uv(const vec3& point) const
    {
        if ((fabsf(pmin.x - point.x) < cc::math::EPS) || (fabsf(pmax.x - point.x) < cc::math::EPS))
        {
            return vec2((point.y - pmin.y) / (pmax.y - pmin.y), (point.z - pmin.z) / (pmax.z - pmin.z));
        }
        if ((fabsf(pmin.y - point.y) < cc::math::EPS) || (fabsf(pmax.y - point.y) < cc::math::EPS))
        {
            return vec2((point.x - pmin.x) / (pmax.x - pmin.x), (point.z - pmin.z) / (pmax.z - pmin.z));
        }
        return vec2((point.x - pmin.x) / (pmax.x - pmin.x), (point.y - pmin.y) / (pmax.y - pmin.y));
    }

    vec3 pmin;
    vec3 pmax;
    IMaterial* mat;
};

