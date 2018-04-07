#include "hitable.hpp"


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

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        const vec3 oc = r.origin() - center;

        const float a = dot(r.direction(), r.direction());
        const float b = dot(oc, r.direction());
        const float c = dot(oc, oc) - radius2;
        const float discriminant = b * b - a * c;

        //
        // b > 0     - ray pointing away from sphere
        // c > 0     - ray does not start inside sphere
        //
        if (c > .0f && b > .0f) return false;

        //
        // discr < 0 - ray does not hit the sphere
        //
        if (discriminant < .0f) return false;

        const float sq_bac = fastsqrt(b * b - a * c);

        float temp = (-b - sq_bac) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.pt(temp);
            rec.normal = (rec.p - center) / radius;
            get_uv_at((rec.p - center) / radius, rec.u, rec.v);
            rec.mat_ptr = mat;

            return true;
        }

        temp = (-b + sq_bac) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.pt(temp);
            rec.normal = (rec.p - center) / radius;
            get_uv_at((rec.p - center) / radius, rec.u, rec.v);
            rec.mat_ptr = mat;

            return true;
        }

        return false;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = aabb(center - vec3(radius), center + vec3(radius));
        return true;
    }

private:
    void get_uv_at(const vec3& p, float& u, float& v) const
    {
        float phi = atan2f(p.z, p.x);
        float theta = asinf(p.y);

        u = 1.0f - (phi + PI) / (2.0f * PI);
        v = (theta + PI / 2.0f) / PI;
    }

    vec3 center;
    float radius;
    float radius2;
    material* mat;
};

