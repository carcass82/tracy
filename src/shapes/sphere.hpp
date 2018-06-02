/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once
#include "hitable.hpp"

using cc::math::vec2;

class sphere : public hitable
{
public:
    sphere(vec3 c, float r, material* m)
        : center(c)
        , radius(r)
        , radius2(r * r)
        , mat(m)
    {
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override final
    {
        vec3 oc = r.origin() - center;

        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius2;
        float discriminant = b * b - a * c;

        //
        // b > 0     - ray pointing away from sphere
        // c > 0     - ray does not start inside sphere
        // discr < 0 - ray does not hit the sphere
        //
        if (!(discriminant < .0f)) {

            float sq_bac = sqrtf(discriminant);

            float temp = (-b - sq_bac) / a;
            if (temp < t_max && temp > t_min) {

                rec.t = temp;
                rec.p = r.pt(temp);
                rec.normal = (rec.p - center) / radius;
                rec.uv = get_uv_at((rec.p - center) / radius);
                rec.mat_ptr = mat;
                return true;

            }

            temp = (-b + sq_bac) / a;
            if (temp < t_max && temp > t_min) {

                rec.t = temp;
                rec.p = r.pt(temp);
                rec.normal = (rec.p - center) / radius;
                rec.uv = get_uv_at((rec.p - center) / radius);
                rec.mat_ptr = mat;
                return true;

            }

        }

        return false;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override final
    {
        box = aabb(center - vec3(radius), center + vec3(radius));
        return true;
    }

    virtual float pdf_value(const vec3& o, const vec3& v) const override final
    {
        hit_record rec;
        if (hit(Ray(o, v), 0.001f, FLT_MAX, rec)) {
            float cos_theta_max = sqrtf(1.f - radius2 / length2(center - o));
            float solid_angle = 2.f * PI * (1.f - cos_theta_max);
            return 1.f / solid_angle;
        }
        return .0f;
    }

    virtual vec3 random(const vec3& o) const override final
    {
        vec3 direction = center - o;
        float d2 = length2(direction);

        mat3 onb = build_orthonormal_basis(direction);
        return random_to_sphere(radius, d2) * onb;
    }

private:
    vec2 get_uv_at(const vec3& p) const
    {
        float phi = atan2f(p.z, p.x);
        float theta = asinf(p.y);

        return vec2{ 1.0f - (phi + PI) / (2.0f * PI), (theta + PI / 2.0f) / PI };
    }

    vec3 center;
    float radius;
    float radius2;
    material* mat;
};

