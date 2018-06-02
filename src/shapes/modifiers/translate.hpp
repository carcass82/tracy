#pragma once
#include "hitable.hpp"

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
            box = aabb(box.vmin + offset, box.vmax + offset);
            return true;
        }
        return false;
    }

    hitable* ptr;
    vec3 offset;
};
