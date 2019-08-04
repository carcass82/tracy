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
		position = eye;

		view = lookAt(eye, center, up);
		projection = perspective(radians(fov), ratio, .1f, 1000.f);

		view_projection_inv = inverse(projection * view);
    }

    Ray get_ray(float s, float t) const
    {
		vec3 pixel_ndc = vec3(s, t, 1.f) * 2.f - 1.f;

		vec4 point_3d = view_projection_inv * vec4(pixel_ndc, 1.f);
		point_3d /= point_3d.w;

		return Ray(position, vec3(point_3d.x, point_3d.y, point_3d.z) - position);
    }

private:
	vec3 position;
	mat4 view;
	mat4 projection;
	mat4 view_projection_inv;
};
