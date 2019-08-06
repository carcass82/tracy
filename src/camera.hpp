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
		position_ = eye;

		view_ = lookAt(eye, center, up);
		projection_ = perspective(radians(fov), ratio, .1f, 1000.f);

		view_projection_inv_ = inverse(projection_ * view_);
    }

    Ray get_ray(float s, float t) const
    {
		vec3 pixel_ndc = vec3(s, t, 1.f) * 2.f - 1.f;

		vec4 point_3d = view_projection_inv_ * vec4(pixel_ndc, 1.f);
		point_3d /= point_3d.w;

		return Ray(position_, vec3(point_3d.x, point_3d.y, point_3d.z) - position_);
    }

private:
	vec3 position_;
	mat4 view_;
	mat4 projection_;
	mat4 view_projection_inv_;
};
