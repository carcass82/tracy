/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */

#pragma once

class NOVTABLE ITexture
{
public:
    virtual vec3 value(const vec2& uv, const vec3& p) const = 0;
};
