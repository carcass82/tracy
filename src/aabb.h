/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "ray.h"

struct BBox
{
    CUDA_DEVICE_CALL BBox(const float in_minbound = FLT_MAX, const float in_maxbound = -FLT_MAX)
        : minbound{ in_minbound }
        , maxbound{ in_maxbound }
    {}

	CUDA_DEVICE_CALL BBox(const vec3& in_minbound, const vec3& in_maxbound)
		: minbound{ in_minbound }
		, maxbound{ in_maxbound }
	{}

    CUDA_DEVICE_CALL vec3 GetCenter() const
    {
        return (minbound + maxbound) * .5f;
    }

    CUDA_DEVICE_CALL vec3 GetSize() const
    {
        return maxbound - minbound;
    }

    CUDA_DEVICE_CALL void Extend(const vec3& point)
    {
        minbound = pmin(minbound, point);
        maxbound = pmax(maxbound, point);
    }

    CUDA_DEVICE_CALL void Extend(const BBox& bbox)
    {
        minbound = pmin(minbound, bbox.minbound);
        maxbound = pmax(maxbound, bbox.maxbound);
    }

    CUDA_DEVICE_CALL void Reset()
    {
        minbound = FLT_MAX;
        maxbound = -FLT_MAX;
    }

	vec3 minbound;
	vec3 maxbound;
};
