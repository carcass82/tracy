/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "ray.h"

class Camera
{
public:
	CUDA_DEVICE_CALL Camera()
	{}

	CUDA_DEVICE_CALL Camera(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		Setup(eye, center, up, fov, ratio);
	}

	CUDA_DEVICE_CALL void Setup(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		eye_ = eye;
		view_ = lookAt(eye, center, up);
		projection_ = perspective(radians(fov), ratio, .1f, 10000.f);
		view_projection_inv_ = inverse(projection_ * view_);
	}

	CUDA_DEVICE_CALL Ray GetRayFrom(float s, float t) const
	{
		vec3 pixel_ndc = vec3(s, t, 1.f) * 2.f - 1.f;
		vec4 point_3d = view_projection_inv_ * vec4(pixel_ndc, 1.f);
		point_3d /= point_3d.w;

		return Ray(eye_, vec3(point_3d.x, point_3d.y, point_3d.z) - eye_);
	}

	CUDA_DEVICE_CALL const mat4& GetView() const { return view_; }

	CUDA_DEVICE_CALL const mat4& GetProjection() const { return projection_; }

private:
	vec3 eye_;
	mat4 view_;
	mat4 projection_;
	mat4 view_projection_inv_;
};
