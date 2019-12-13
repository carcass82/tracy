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
    CUDA_CALL Ray()
        : origin_{}
        , direction_{}
        , inv_direction_{}
    {}

    CUDA_CALL Ray(const vec3& origin, const vec3& direction)
        : origin_(origin)
        , direction_(normalize(direction))
		, inv_direction_(1.f / direction_)
    {}

    CUDA_CALL constexpr    const vec3& GetOrigin() const        { return origin_; }
    CUDA_CALL constexpr    const vec3& GetDirection() const     { return direction_; }
	CUDA_CALL constexpr    const vec3& GetInvDirection() const  { return inv_direction_; }
    CUDA_CALL CC_CONSTEXPR const vec3  GetPoint(float t) const  { return origin_ + t * direction_; }

private:
    vec3 origin_;
    vec3 direction_;
	vec3 inv_direction_;
};
