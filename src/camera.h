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
	Camera()
	{}

	Camera(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		Setup(eye, center, up, fov, ratio);
	}

	void Setup(const vec3& eye, const vec3& center, const vec3& up, float fov, float ratio)
	{
		UpdateProjection(fov, ratio);
		UpdateView(eye, center, up);
	}

	CUDA_ANY Ray GetRayFrom(float s, float t) const
	{
		vec3 pixel_ndc = vec3(s, t, 1.f) * 2.f - 1.f;
		vec4 point_3d = view_projection_inv_ * vec4(pixel_ndc, 1.f);
		point_3d /= point_3d.w;

		return Ray(eye_, normalize(vec3(point_3d.x, point_3d.y, point_3d.z) - eye_));
	}

	void UpdateProjection(float fov, float ratio, float znear = .1f, float zfar = 10000.f)
	{
		fov_ = fov;
		ratio_ = ratio;
		near_far_ = { znear, zfar };

		projection_ = perspective(radians(fov_), ratio_, near_far_.x, near_far_.y);
		view_projection_inv_ = inverse(projection_ * view_);
	}

	void UpdateView(const vec3& eye, const vec3& center, const vec3& up)
	{
		eye_ = eye;
		center_ = center;
		up_ = up;

		view_ = lookAt(eye_, center_, up_);
		view_projection_inv_ = inverse(projection_ * view_);
	}

	constexpr const mat4& GetView() const { return view_; }

	constexpr const mat4& GetProjection() const { return projection_; }

	constexpr const vec3& GetPosition() const { return eye_; }

	constexpr const vec3& GetUpVector() const { return up_; }

	constexpr const vec3& GetTarget() const { return center_; }

	constexpr vec3 GetViewDirection() const { return center_ - eye_; }

	void BeginFrame() const {}
	
	void EndFrame() const {}

private:
	vec3 eye_{};
	vec3 center_{};
	vec3 up_{};

	float fov_{};
	float ratio_{};
	vec2 near_far_{};

	mat4 view_{};
	mat4 projection_{};
	mat4 view_projection_inv_{};
};
