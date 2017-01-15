/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

class ray
{
public:
    ray()
        : m_origin()
        , m_direction()
    {
    }
    
    ray(const glm::vec3& origin, const glm::vec3& direction)
        : m_origin(origin)
        , m_direction(direction)
    {
    }

    const glm::vec3& origin() const
    {
        return m_origin;
    }
    
    const glm::vec3& direction() const
    {
        return m_direction;
    }
    
    glm::vec3 point_at_parameter(float t) const
    {
        return m_origin + t * m_direction;
    }

private:
    glm::vec3 m_origin;
    glm::vec3 m_direction;
};
