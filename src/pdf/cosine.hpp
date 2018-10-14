#pragma once
#include "pdf.hpp"
#include "geom.hpp"

class cosine_pdf : public pdf
{
public:
    cosine_pdf(const vec3& w) { uvw = build_orthonormal_basis(w); }

    float value(const vec3 &direction) const override final
    {
        float cosine = dot(normalize(direction), uvw[2]);
        return (cosine > .0f)? cosine / PI : .0f;
    }

    vec3 generate() const override final
    {
        return random_cosine_GetDirection() * uvw;
    }

private:
    mat3 uvw;
};
