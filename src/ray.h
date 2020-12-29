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
    constexpr Ray()
    {}

    constexpr Ray(const vec3& origin, const vec3& direction)
        : origin_{ origin }
        , direction_{ direction }
        , inv_direction_{ rcp(direction_) }
    {}

    // make sure we don't copy rays around
    Ray(const Ray&) = delete;
    Ray& operator=(const Ray&) = delete;

    constexpr Ray(Ray&& other) noexcept
        : origin_{ std::move(other.origin_) }
        , direction_{ std::move(other.direction_) }
        , inv_direction_{ std::move(other.inv_direction_) }
    {}

    constexpr Ray& operator=(Ray&& other) noexcept
    {
        if (this != &other)
        {
            origin_ = std::move(other.origin_);
            direction_ = std::move(other.direction_);
            inv_direction_ = std::move(other.inv_direction_);
        }

        return *this;
    }

    constexpr const vec3& GetOrigin() const            { return origin_; }
    constexpr const vec3& GetDirection() const         { return direction_; }
	constexpr const vec3& GetDirectionInverse() const  { return inv_direction_; }
    constexpr const vec3  GetPoint(float t) const      { return origin_ + t * direction_; }

private:
    vec3 origin_{};
    vec3 direction_{};
    vec3 inv_direction_{};
};
