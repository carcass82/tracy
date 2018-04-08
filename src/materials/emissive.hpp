#pragma once
#include "material.hpp"

class emissive : public material
{
public:
    emissive(texture* a)
        : emit(a)
    {
    }

    virtual bool scatter(const Ray& r_in, const hit_record& rec, vec3& attenuation, Ray& scattered) const override
    {
        return false;
    }

    virtual vec3 emitted(const vec2& uv, const vec3& p) const override
    {
        return emit->value(uv, p);
    }

private:
    texture* emit;
};

