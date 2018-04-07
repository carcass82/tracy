/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once
#include "tmath.h"
#include "ray.hpp"

using vmath::vec3;


class aabb
{
public:
    aabb()
        : m_min()
        , m_max()
    {
    }

    aabb(const vec3& a, const vec3& b)
        : m_min(a)
        , m_max(b)
    {
    }

    void expand(const aabb& other_box)
    {
        m_min.x = vutil::min(m_min.x, other_box.min().x);
        m_min.y = vutil::min(m_min.y, other_box.min().y);
        m_min.z = vutil::min(m_min.z, other_box.min().z);

        m_max.x = vutil::max(m_max.x, other_box.max().x);
        m_max.y = vutil::max(m_max.y, other_box.max().y);
        m_max.z = vutil::max(m_max.z, other_box.max().z);
    }

    const vec3& min() const
    {
        return m_min;
    }

    const vec3& max() const
    {
        return m_max;
    }

    bool hit(const Ray& r, float tmin, float tmax) const
    {
        for (int i = 0; i < 3; ++i) {

            float a = (m_min[i] - r.origin()[i]) / r.direction()[i];
            float b = (m_max[i] - r.origin()[i]) / r.direction()[i];

            if (vutil::min(vutil::max(a, b), tmax) <= vutil::max(vutil::min(a, b), tmin))
                return false;
        }

        return true;
    }

private:
    vec3 m_min;
    vec3 m_max;
};
