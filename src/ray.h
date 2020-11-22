/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

class Ray
{
public:
    CUDA_DEVICE_CALL Ray()
    {}

    CUDA_DEVICE_CALL Ray(const vec3& origin, const vec3& direction)
        : origin_{ origin }
        , direction_{ normalize(direction) }
        , inv_direction_{ 1.f / direction_ }
    {}

    // make sure we don't copy rays around
    Ray(const Ray&) = delete;
    Ray& operator=(const Ray&) = delete;

    CUDA_DEVICE_CALL Ray(Ray&& other)
        : Ray(other.origin_, other.direction_)
    {}

    CUDA_DEVICE_CALL Ray& operator=(Ray&& other) noexcept
    {
        if (this != &other)
        {
            origin_ = std::move(other.origin_);
            direction_ = std::move(other.direction_);
            inv_direction_ = std::move(other.inv_direction_);
        }

        return *this;
    }

    CUDA_DEVICE_CALL constexpr const vec3& GetOrigin() const            { return origin_; }
    CUDA_DEVICE_CALL constexpr const vec3& GetDirection() const         { return direction_; }
	CUDA_DEVICE_CALL constexpr const vec3& GetDirectionInverse() const  { return inv_direction_; }
    CUDA_DEVICE_CALL constexpr const vec3  GetPoint(float t) const      { return origin_ + t * direction_; }

private:
    vec3 origin_{};
    vec3 direction_{};
    vec3 inv_direction_{};
};
