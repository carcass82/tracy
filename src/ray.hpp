/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

//
// a ray represented in its parametric form
// A - ray origin
// B - ray direction
//
class Ray
{
public:
    Ray() {}

    Ray(const vec3& a, const vec3& b)
        : origin_(a)
        , direction_(b)
    {
    }

    constexpr const vec3& get_origin() const     { return origin_; }
    constexpr const vec3& get_direction() const  { return direction_; }
    vec3 point_at(float t) const                 { return origin_ + t * direction_; }

private:
    vec3 origin_;
    vec3 direction_;
};
