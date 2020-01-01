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

//
// https://tavianator.com/fast-branchless-raybounding-box-intersections/
//
CUDA_DEVICE_CALL inline bool IntersectsWithBoundingBox(const BBox& box, const Ray& ray, float nearest_intersection = FLT_MAX)
{
    const vec3 minbound = (box.minbound - ray.GetOrigin()) * ray.GetInvDirection();
    const vec3 maxbound = (box.maxbound - ray.GetOrigin()) * ray.GetInvDirection();

    const vec3 tmin1 = pmin(minbound, maxbound);
    const vec3 tmax1 = pmax(minbound, maxbound);

    const float tmin = max(tmin1.x, max(tmin1.y, tmin1.z));
    const float tmax = min(tmax1.x, min(tmax1.y, tmax1.z));

    return (tmax >= max(1.e-8f, tmin) && tmin < nearest_intersection);
}
