/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/norm.hpp"

#include "aabb.hpp"

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

void get_sphere_uv(const vec3& p, float& u, float& v)
{
	float phi = atan2(p.z, p.x);
	float theta = glm::asin(p.y);
	u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
	v = (theta + M_PI / 2.0f) / M_PI;
}

float schlick(float cos, float ref_idx)
{
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1 - cos), 5);
}

aabb surrounding_box(const aabb& box0, const aabb& box1)
{
	vec3 small(glm::min(box0.min().x, box1.min().x), glm::min(box0.min().y, box1.min().y), glm::min(box0.min().z, box1.min().z));
	vec3 big(glm::max(box0.max().x, box1.max().x), glm::max(box0.max().y, box1.max().y), glm::max(box0.max().z, box1.max().z));

	return aabb(small, big);
}

float trilinear_interp(float c[2][2][2], float u, float v, float w)
{
	float accum = 0.0f;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) * (k * w + (1 - k) * (1 -w)) * c[i][j][k];

	return accum;
}

float perlin_interp(vec3 c[2][2][2], float u, float v, float w)
{
	// hermite cubic
	float uu = u * u * (3 - 2 * u);
	float vv = v * v * (3 - 2 * v);
	float ww = w * w * (3 - 2 * w);
	float accum = 0.0f;	
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				vec3 weight_v(u - i, v - j, w - k);
				accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
			}
		}
	}

	return accum;
}
