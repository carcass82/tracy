/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2017-2021
 */
#pragma once
#include "common.h"
#include "ray.h"

struct BBox
{
    BBox(const float in_minbound = FLT_MAX, const float in_maxbound = -FLT_MAX)
        : minbound{ in_minbound }
        , maxbound{ in_maxbound }
    {}

	BBox(const vec3& in_minbound, const vec3& in_maxbound)
		: minbound{ in_minbound }
		, maxbound{ in_maxbound }
	{}

    vec3 GetCenter() const
    {
        return (minbound + maxbound) * .5f;
    }

    vec3 GetSize() const
    {
        return maxbound - minbound;
    }

    bool Contains(const vec3& point) const
    {
        return point.x >= minbound.x && point.x <= maxbound.x &&
               point.y >= minbound.y && point.y <= maxbound.y &&
               point.z >= minbound.z && point.z <= maxbound.z;
    }

    void Extend(const vec3& point)
    {
        minbound = pmin(minbound, point);
        maxbound = pmax(maxbound, point);
    }

    void Extend(const BBox& bbox)
    {
        minbound = pmin(minbound, bbox.minbound);
        maxbound = pmax(maxbound, bbox.maxbound);
    }

    void Reset()
    {
        minbound = vec3(FLT_MAX);
        maxbound = vec3(-FLT_MAX);
    }

	vec3 minbound;
	vec3 maxbound;
};
