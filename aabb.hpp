/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

#include "ray.hpp"

class aabb
{
public:
    aabb()
        : m_min()
        , m_max()
    {
    }

    aabb(const glm::vec3& a, const glm::vec3& b)
        : m_min(a)
        , m_max(b)
    {
    }

    const glm::vec3& min() const 
    {
        return m_min;
    }
    
    const glm::vec3& max() const
    {
        return m_max;
    }

    bool hit(const ray& r, float tmin, float tmax) const
    {
        for (int i = 0; i < 3; ++i) {
            
            float a = (m_min[i] - r.origin()[i]) / r.direction()[i];
            float b = (m_max[i] - r.origin()[i]) / r.direction()[i];
            
            if (glm::min(glm::max(a, b), tmax) <= glm::max(glm::min(a, b), tmin))
                return false;
        }

        return true;
    }

private:
    glm::vec3 m_min;
    glm::vec3 m_max;
};
