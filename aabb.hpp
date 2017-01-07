/*
 * Tracy, a simple raytracer
 * inspired by Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

#include "ray.hpp"

class aabb
{
public:
	aabb() {}
	aabb(const vec3& a, const vec3& b)
		: _min(a), _max(b) {}

	const vec3& min() const { return _min; }
	const vec3& max() const { return _max; }

	bool hit(const ray& r, float tmin, float tmax) const
	{
		for (int a = 0; a < 3; ++a) {
			float t0 = glm::min((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);
			float t1 = glm::max((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);

			if (glm::min(t1, tmax) <= glm::max(t0, tmin))
				return false;
		}
		return true;
	}


	vec3 _min;
	vec3 _max;
};
