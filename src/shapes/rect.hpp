#pragma once
#include "hitable.hpp"

class xy_rect : public hitable
{
public:
    xy_rect() {}
    xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat)
        : x0(_x0)
        , x1(_x1)
        , y0(_y0)
        , y1(_y1)
        , k(_k)
        , mp(mat)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        float t = (k - r.origin().z) / r.direction().z;
        if (t < t_min || t > t_max)
            return false;

        float x = r.origin().x + t * r.direction().x;
        float y = r.origin().y + t * r.direction().y;
        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;

        rec.t = t;
        rec.uv = { (x - x0) / (x1 - x0), (y - y0) / (y1 - y0) };
        rec.mat_ptr = mp;
        rec.p = r.pt(t);
        rec.normal = vec3(0, 0, 1);
        return true;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = aabb(vec3(x0, y0, k - 0.0001f), vec3(x1, y1, k + 0.0001f));
        return true;
    }

    float x0;
    float x1;
    float y0;
    float y1;
    float k;
    material* mp;
};


class xz_rect : public hitable
{
public:
    xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat)
        : x0(_x0)
        , x1(_x1)
        , z0(_z0)
        , z1(_z1)
        , k(_k)
        , mp(mat)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        float t = (k - r.origin().y) / r.direction().y;
        if (t < t_min || t > t_max)
            return false;

        float x = r.origin().x + t * r.direction().x;
        float z = r.origin().z + t * r.direction().z;
        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;

        rec.uv = { (x - x0) / (x1 - x0), (z - z0) / (z1 - z0) };
        rec.t = t;
        rec.mat_ptr = mp;
        rec.p = r.pt(t);
        rec.normal = vec3(0, 0, 1);
        return true;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = aabb(vec3(x0, k - 0.0001f, z0), vec3(x1, k + 0.0001f, z1));
        return true;
    }

    float x0;
    float x1;
    float z0;
    float z1;
    float k;
    material* mp;
};


class yz_rect : public hitable
{
public:
    yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat)
        : y0(_y0)
        , y1(_y1)
        , z0(_z0)
        , z1(_z1)
        , k(_k)
        , mp(mat)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        float t = (k - r.origin().x) / r.direction().x;
        if (t < t_min || t > t_max)
            return false;

        float y = r.origin().y + t * r.direction().y;
        float z = r.origin().z + t * r.direction().z;
        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;

        rec.uv = { (y - y0) / (y1 - y0), (z - z0) / (z1 - z0) };
        rec.t = t;
        rec.mat_ptr = mp;
        rec.p = r.pt(t);
        rec.normal = vec3(0, 0, 1);
        return true;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = aabb(vec3(k - 0.0001f, y0, z0), vec3(k + 0.0001f, y1, z1));
        return true;
    }

    float y0;
    float y1;
    float z0;
    float z1;
    float k;
    material* mp;
};

