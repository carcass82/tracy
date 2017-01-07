#pragma once

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

vec3 random_in_unit_sphere()
{
	vec3 p;
	do {
		p = 2.0f * vec3(drand48(), drand48(), drand48()) - vec3(1.0f, 1.0f, 1.0f);
	} while (length2(p) >= 1.0f);

	return p;
}

vec3 random_in_unit_disk()
{
	vec3 p;
	do {
		p = 2.0f * vec3(drand48(), drand48(), 0) - vec3(1.0f, 1.0f, 0.0f);
	} while (dot(p, p) >= 1.0f);

	return p;
}

float schlick(float cos, float ref_idx)
{
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1 - cos), 5);
}
