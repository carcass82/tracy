/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "tmath.h"
using vmath::vec3;
using vmath::radians;
using vmath::cross;
using vmath::normalize;

class camera
{
public:
    camera()
    {
    }

    camera(const vec3& lookfrom, const vec3& lookat, const vec3& vup, float vfov, float aspect, float aperture, float focus_dist)
    {
        setup(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }

    void setup(const vec3& lookfrom, const vec3& lookat, const vec3& vup, float vfov, float aspect, float aperture, float focus_dist)
    {
        lens_radius = aperture / 2.0f;

        float theta = radians(vfov);
        float half_height = tanf(theta / 2.0f);
        float half_width = aspect * half_height;

        origin = lookfrom;
        vec3 w = normalize(lookfrom - lookat);
        vec3 u = normalize(cross(vup, w));
        vec3 v = cross(w, u);

        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    Ray get_ray(float s, float t)
    {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x + v * rd.y;
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
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
