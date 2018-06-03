#pragma once
#include "pdf.hpp"
#include "hitable.hpp"

class hitable_pdf : public pdf
{
public:
    hitable_pdf(hitable* p, const vec3& origin) : ptr(p), o(origin) {}

    float value(const vec3& direction) const override final { return ptr->pdf_value(o, direction); }
    vec3 generate() const override final { return ptr->random(o); }

private:
    hitable* ptr;
    vec3 o;
};
