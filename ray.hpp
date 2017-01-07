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
	ray()                                          {}
	ray(const vec3& a, const vec3& b, float t)
		: A(a), B(b), _time(t)                     {}

	vec3 origin() const                            { return A; }
	vec3 direction() const                         { return B; }
	float time() const                             { return _time; }
	vec3 point_at_parameter(float t) const         { return A + t * B; }

	vec3 A;
	vec3 B;
	float _time;
};
