/*
 * Tracy, a simple raytracer
 * inspired by Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

class camera
{
public:
	camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist)
	{
		lens_radius = aperture / 2.0f;

		float theta = vfov * M_PI / 180.0f;
		float half_height = tan(theta / 2.0f);
		float half_width = aspect * half_height;

		origin = lookfrom;
		vec3 w = normalize(lookfrom - lookat);
		vec3 u = normalize(cross(vup, w));
		vec3 v = cross(w, u);

		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		horizontal = 2.0f * half_width * focus_dist * u;
		vertical = 2.0f * half_height * focus_dist * v;
	}

	ray get_ray(float s, float t)
	{
		vec3 rd = lens_radius * random_in_unit_disk();
		vec3 offset = u * rd.x + v * rd.y;

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u;
	vec3 v;
	vec3 w;
	float lens_radius;
};
