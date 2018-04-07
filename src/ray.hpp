/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec3;

class ray
{
public:
    ray()
        : m_origin()
        , m_direction()
    {
    }

    ray(const vec3& origin, const vec3& direction)
        : m_origin(origin)
        , m_direction(direction)
    {
    }

    const vec3& origin() const
    {
        return m_origin;
    }

    const vec3& direction() const
    {
        return m_direction;
    }

    vec3 point_at_parameter(float t) const
    {
        return m_origin + t * m_direction;
    }

private:
    vec3 m_origin;
    vec3 m_direction;
};
