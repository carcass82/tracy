#pragma once
#include "hitable.hpp"

class rotate_y : public hitable
{
public:
    rotate_y(hitable* p, float angle)
        :ptr(p)
    {
        float anglerad = radians(angle);
        sin_theta = sin(anglerad);
        cos_theta = cos(anglerad);
        hasbox = ptr->bounding_box(0, 1, bbox);

        vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    float x = i * bbox.vmax.x + (1 - i) * bbox.vmin.x;
                    float y = j * bbox.vmax.y + (1 - j) * bbox.vmin.y;
                    float z = k * bbox.vmax.z + (1 - k) * bbox.vmin.z;
                    float newx = cos_theta * x + sin_theta * z;
                    float newz = -sin_theta * x + cos_theta * z;

                    vec3 tester(newx, y, newz);
                    for (int c = 0; c < 3; ++c) {
                        if (tester[c] > max[c]) max[c] = tester[c];
                        if (tester[c] < min[c]) min[c] = tester[c];
                    }
                }
            }
        }

        bbox = aabb(min, max);
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override
    {
        vec3 origin = r.origin();
        vec3 direction = r.direction();

        origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
        origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
        direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
        direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

        Ray rotated_r(origin, direction);

        if (ptr->hit(rotated_r, t_min, t_max, rec)) {
            vec3 p = rec.p;
            vec3 normal = rec.normal;

            p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
            p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

            normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
            normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

            rec.p = p;
            rec.normal = normal;
            return true;
        }

        return false;
    }

    virtual bool bounding_box(float t0, float t1, aabb& box) const override
    {
        box = bbox;
        return hasbox;
    }

    hitable* ptr;
    bool hasbox;
    aabb bbox;
    float sin_theta;
    float cos_theta;
};
