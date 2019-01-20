/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

class Camera
{
public:
    Camera() {}

    Camera(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
    {
        setup(eye, center, up, fov, ratio);
    }

    void setup(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
    {
        float theta = radians(fov);
        float half_height = tanf(theta / 2.0f);
        float half_width = ratio * half_height;

		position = eye;
        vec3 w = normalize(eye - center);
        vec3 u = normalize(cross(up, w));
        vec3 v = cross(w, u);

        lower_left_corner = eye - half_width * u - half_height * v - w;
        horizontal = 2.0f * half_width * u;
        vertical = 2.0f * half_height * v;
    }

    Ray get_ray(float s, float t) const
    {
        return Ray(position, normalize(lower_left_corner + s * horizontal + t * vertical - position));
    }

private:
	vec3 position;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};
