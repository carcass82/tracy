/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

class camera
{
public:
    camera()
    {
    }
    
    camera(const glm::vec3& lookfrom, const glm::vec3& lookat, const glm::vec3& vup, float vfov, float aspect, float aperture, float focus_dist)
    {
        setup(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }
    
    void setup(const glm::vec3& lookfrom, const glm::vec3& lookat, const glm::vec3& vup, float vfov, float aspect, float aperture, float focus_dist)
    {
        lens_radius = aperture / 2.0f;

        float theta = glm::radians(vfov);
        float half_height = glm::tan(theta / 2.0f);
        float half_width = aspect * half_height;

        origin = lookfrom;
        glm::vec3 w = normalize(lookfrom - lookat);
        glm::vec3 u = normalize(cross(vup, w));
        glm::vec3 v = cross(w, u);

        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    ray get_ray(float s, float t)
    {
        glm::vec3 rd = lens_radius * random_in_unit_disk();
        glm::vec3 offset = u * rd.x + v * rd.y;
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    glm::vec3 origin;
    glm::vec3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u;
    glm::vec3 v;
    glm::vec3 w;
    float lens_radius;
};
