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
	CUDA_CALL Camera()
	{}

	CUDA_CALL Camera(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		Setup(eye, center, up, fov, ratio);
	}

	CUDA_CALL void Setup(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		eye_ = eye;
		view_ = lookAt(eye, center, up);
		projection_ = perspective(radians(fov), ratio, .1f, 10000.f);
		view_projection_inv_ = inverse(projection_ * view_);
	}

	CUDA_CALL Ray GetRayFrom(float s, float t) const
	{
		vec3 pixel_ndc = vec3(s, t, 1.f) * 2.f - 1.f;
		vec4 point_3d = view_projection_inv_ * vec4(pixel_ndc, 1.f);
		point_3d /= point_3d.w;

		return Ray(eye_, vec3(point_3d.x, point_3d.y, point_3d.z) - eye_);
	}

	CUDA_CALL const mat4& GetView() const { return view_; }

	CUDA_CALL const mat4& GetProjection() const { return projection_; }

	void Translate(const vec3& dir)
	{
		view_ = translate(view_, dir);
		view_projection_inv_ = inverse(projection_ * view_);
		dirty_ = true;
	}

	void Rotate(const vec3& angle)
	{
		view_ = rotate(view_, radians(angle.x), vec3{ 1.0f,  .0f, .0f });
		view_ = rotate(view_, radians(angle.y), vec3{ .0f, 1.0f, .0f });
		view_projection_inv_ = inverse(projection_ * view_);
		dirty_ = true;
	}

	bool IsDirty() const
	{
		return dirty_;
	}

	void SetDirty(bool dirty)
	{
		dirty_ = dirty;
	}

private:
	vec3 eye_;
	bool dirty_;
	mat4 view_;
	mat4 projection_;
	mat4 view_projection_inv_;
};
