/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <iostream>
#include <algorithm>
#include "tmath.h"
#include "geom.hpp"
#include "ray.hpp"
#include "aabb.hpp"
#include "material.hpp"
#include "texture.hpp"

using vutil::max;
using vmath::radians;
using vmath::PI;
using vmath::fastsqrt;

class material;
class isotropic;

class hitable
{
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};




class flip_normals : public hitable
{
public:
    flip_normals(hitable* p)
        : ptr(p)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        if (ptr->hit(r, t_min, t_max, rec)) {
            rec.normal = rec.normal * -1;
            return true;
        } else {
            return false;
        }
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        return ptr->bounding_box(t0, t1, box);
    }

    hitable* ptr;
};


class translate : public hitable
{
public:
    translate(hitable* p, const vec3& displacement)
        : ptr(p)
        , offset(displacement)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        Ray moved_r(r.origin() - offset, r.direction());
        if (ptr->hit(moved_r, t_min, t_max, rec)) {
            rec.p += offset;
            return true;
        }
        return false;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        if (ptr->bounding_box(t0, t1, box)) {
            box = aabb(box.min() + offset, box.max() + offset);
            return true;
        }
        return false;
    }

    hitable* ptr;
    vec3 offset;
};


class rotate_y : public hitable
{
public:
    rotate_y(hitable* p, float angle)
        :ptr(p)
    {
        float anglerad = radians(angle);
        sin_theta = sin(anglerad);
        cos_theta = cos(anglerad);
        hasbox = ptr->bounding_box(0, 1, bbox);

        vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    float x = i * bbox.max().x + (1 - i) * bbox.min().x;
                    float y = j * bbox.max().y + (1 - j) * bbox.min().y;
                    float z = k * bbox.max().z + (1 - k) * bbox.min().z;
                    float newx = cos_theta * x + sin_theta * z;
                    float newz = -sin_theta * x + cos_theta * z;

                    vec3 tester(newx, y, newz);
                    for (int c = 0; c < 3; ++c) {
                        if (tester[c] > max[c]) max[c] = tester[c];
                        if (tester[c] < min[c]) min[c] = tester[c];
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        vec3 origin = r.origin();
        vec3 direction = r.direction();

        origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
        origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
        direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
        direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

        Ray rotated_r(origin, direction);

        if (ptr->hit(rotated_r, t_min, t_max, rec)) {
            vec3 p = rec.p;
            vec3 normal = rec.normal;

            p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
            p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

            normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
            normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

            rec.p = p;
            rec.normal = normal;
            return true;
        }

        return false;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = bbox;
        return hasbox;
    }

    hitable* ptr;
    bool hasbox;
    aabb bbox;
    float sin_theta;
    float cos_theta;
};


class hitable_list : public hitable
{
public:
    hitable_list() {}
    hitable_list(hitable** l, int n)
        : list(l)
        , list_size(n)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;

        float closest_so_far = t_max;
        for (int i = 0; i < list_size; ++i) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        if (list_size < 1)
            return false;

        aabb temp_box;
        bool first_true = list[0]->bounding_box(t0, t1, temp_box);
        if (!first_true)
            return false;
        else
            box = temp_box;

        for (int i = 1; i < list_size; ++i) {
            if (list[0]->bounding_box(t0, t1, temp_box)) {
                box.expand(temp_box);
            } else {
                return false;
            }
        }
        return true;
    }

    hitable** list;
    int list_size;
};
