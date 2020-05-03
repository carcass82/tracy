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
	CUDA_DEVICE_CALL BBox()
		: minbound{}
		, maxbound{}
	{}

    CUDA_DEVICE_CALL BBox(const float in_minbound, const float in_maxbound)
        : minbound(in_minbound, in_minbound, in_minbound)
        , maxbound(in_maxbound, in_maxbound, in_maxbound)
    {}

	CUDA_DEVICE_CALL BBox(const vec3& in_minbound, const vec3& in_maxbound)
		: minbound(in_minbound)
		, maxbound(in_maxbound)
	{}

    CUDA_DEVICE_CALL vec3 GetCenter() const
    {
        return (minbound + maxbound) / 2.f;
    }

    CUDA_DEVICE_CALL vec3 GetSize() const
    {
        return maxbound - minbound;
    }

	vec3 minbound;
	vec3 maxbound;
};
