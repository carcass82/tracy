/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "noise.hpp"

class texture
{
public:
	virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public texture
{
public:
	constant_texture() {}
	constant_texture(const vec3& c) : color(c) {}

	virtual vec3 value(float u, float v, const vec3& p) const override
	{
		return color;
	}

	vec3 color;
};

class checker_texture : public texture
{
public:
	checker_texture() {}
	checker_texture(texture* t0, texture* t1): even(t0), odd(t1) {}

	virtual vec3 value(float u, float v, const vec3& p) const override
	{
		float sines = glm::sin(10 * p.x) * glm::sin(10 * p.y) * sin(10 * p.z);
		return (sines < 0.0f)? odd->value(u, v, p) : even->value(u, v, p);
	}

	texture* odd;
	texture* even;
};

class noise_texture : public texture
{
public:
	noise_texture() {}
	noise_texture(float sc) : scale(sc) {}

	virtual vec3 value(float u, float v, const vec3& p) const override
	{
		//return vec3(1.0f) * noise.noise(p * scale);
		return vec3(1.0f) * 0.5f * (1.0f + glm::sin(scale * p.z + 10 * noise.turb(p)));
	}

	perlin noise;
	float scale;
};
