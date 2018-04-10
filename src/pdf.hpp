/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec3;
using vmath::mat3;

//
// probability density function
//
class pdf
{
public:
    virtual ~pdf() {}
    virtual float value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf
{
public:
    cosine_pdf(const vec3& w) { uvw = build_orthonormal_basis(w); }

    float value(const vmath::vec3 &direction) const override final
    {
        float cosine = dot(normalize(direction), uvw[2]);
        return (cosine > .0f)? cosine / PI : .0f;
    }

    vec3 generate() const override final
    {
        return random_cosine_direction() * uvw;
    }

private:
    mat3 uvw;
};

class hitable_pdf : public pdf
{
public:
    hitable_pdf(hitable* p, const vec3& origin) : ptr(p), o(origin) {}

    float value(const vmath::vec3 &direction) const override final { return ptr->pdf_value(o, direction); }
    vec3 generate() const override final { return ptr->random(o); }

private:
    hitable* ptr;
    vec3 o;
};

class mixture_pdf : public pdf
{
public:
    mixture_pdf(pdf* p0, pdf* p1, float prob1) : p{p0, p1}, prob(prob1) { }

    float value(const vmath::vec3 &direction) const override final { return prob * p[0]->value(direction) + (1.f - prob) * p[1]->value(direction); }
    vec3 generate() const override final { return (fastrand() < prob)? p[0]->generate() : p[1]->generate(); }

private:
    pdf* p[2];
    float prob;
};


class custom_pdf : public pdf
{
public:
    float value(const vmath::vec3 &direction) const override final { return .5f; }
    vec3 generate() const override final { return vec3{1, 0, 0}; }

    //
    // avoid memleaks or simply out-of-memory for too much new PDFs
    // (at the cost of objects created and destoyed each loop - profiler does seem happy though!)
    //
    static void generate_all(hitable* shape, const vec3& o, const vec3& w, Ray& out_scattered, float& out_pdf)
    {
        hitable_pdf lightpdf(shape, o);
        cosine_pdf cospdf(w);

        mixture_pdf mixpdf(&cospdf, &lightpdf, .45f);

        out_scattered = Ray(o, mixpdf.generate());
        out_pdf = mixpdf.value(out_scattered.direction());
    }
};
