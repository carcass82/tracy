/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "hitable.hpp"

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

    virtual float pdf_value(const vec3& o, const vec3& v) const override final
    {
        float w = 1.f / list_size;
        float sum = .0f;
        for (int i = 0; i < list_size; ++i) {
            sum += w * list[i]->pdf_value(o, v);
        }
        return sum;

        //return list[int(fastrand() * list_size)]->pdf_value(o, v);
        //return list[0]->pdf_value(o, v);;
    }

    virtual vec3 random(const vec3& o) const override final
    {
        return list[int(fastrand() * list_size)]->random(o);
        //return list[0]->random(o);
    }

    hitable** list;
    int list_size;
};

