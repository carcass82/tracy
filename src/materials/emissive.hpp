#pragma once
#include "material.hpp"

class emissive : public material
{
public:
    emissive(texture* a)
        : emit(a)
    {
    }

    virtual vec3 emitted(const Ray& r_in, const hit_record& rec, const vec2& uv, const vec3& p) const override
    {
        if (dot(rec.normal, r_in.direction()) < .0f)
            return emit->value(uv, p);
        else
            return vec3{0, 0, 0};
    }

private:
    texture* emit;
};

