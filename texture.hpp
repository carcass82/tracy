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


class image_texture : public texture
{
public:
	image_texture() {}
	image_texture(unsigned char* pixels, int A, int B)
		: data(pixels), nx(A), ny(B) {}

	virtual vec3 value(float u, float v, const vec3& p) const override
	{
		//int i = glm::clamp(int(u * nx),                 0, nx - 1);
		//int j = glm::clamp(int(1.0f - v * ny - 0.001f), 0, ny - 1);

		int i = (u) * nx;
		int j = (1-v) * ny - 0.001;
		if (i < 0) i = 0;
		if (j < 0) j = 0;
		if (i > nx - 1) i = nx - 1;
		if (j > ny - 1) j = ny - 1;

		float r = int(data[3 * i + 3 * nx * j + 0]) / 255.0f;
		float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0f;
		float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0f;

		return vec3(r, g, b);
	}

	unsigned char* data;
	int nx;
	int ny;
};
