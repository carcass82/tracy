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
		
		projection = perspective(radians(fov), ratio, .1f, 100.f);

		projection_inv = inverse(projection);
    }

    Ray get_ray(float s, float t) const
    {
		vec4 pixel(s, t, 1.f, 1.f);
		vec4 pixel_ndc = pixel * 2.f - 1.f;

		vec4 point_3d = projection_inv * pixel_ndc;
		point_3d /= point_3d.w;

		return Ray(position, vec3(point_3d.x, point_3d.y, point_3d.z));
    }

private:
	vec3 position;
	mat4 view;
	mat4 projection;
	mat4 projection_inv;
};
