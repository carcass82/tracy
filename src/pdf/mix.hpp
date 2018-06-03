#pragma once
#include "pdf.hpp"
#include "geom.hpp"

class mix_pdf : public pdf
{
public:
    mix_pdf(pdf* p0, pdf* p1, float prob1) : p{p0, p1}, prob(prob1) { }

    float value(const vec3& direction) const override final { return prob * p[0]->value(direction) + (1.f - prob) * p[1]->value(direction); }
    vec3 generate() const override final { return (fastrand() < prob)? p[0]->generate() : p[1]->generate(); }

private:
    pdf* p[2];
    float prob;
};

