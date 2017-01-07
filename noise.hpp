/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include "glm/glm.hpp"

#include "geom.hpp"

class perlin
{
public:
	float noise(const vec3& p) const
	{
		float u = p.x - floor(p.x);
		float v = p.y - floor(p.y);
		float w = p.z - floor(p.z);

		int i = floor(p.x);
		int j = floor(p.y);
		int k = floor(p.z);

		vec3 c[2][2][2];
		for (int di = 0; di < 2; ++di)
			for (int dj = 0; dj < 2; ++dj)
				for (int dk = 0; dk < 2; ++dk)
					c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];

		return perlin_interp(c, u, v, w);
	}

	float turb(const vec3& p, int depth = 7) const
	{
		float accum = 0.0f;
		vec3 temp_p = p;
		float weight = 1.0f;

		for (int i = 0; i < depth; ++i) {
			accum += weight * noise(temp_p);
			weight *= 0.5f;
			temp_p *= 2.0f;
		}

		return fabsf(accum);
	}

	static vec3* ranvec;
	static int* perm_x;
	static int* perm_y;
	static int* perm_z;
};

static vec3* perlin_generate()
{
	vec3* p = new vec3[256];
	for (int i = 0; i < 256; ++i)
		p[i] = normalize(vec3(-1 + 2 * drand48()));

	return p;
}

void permute(int* p, int n)
{
	for (int i = n - 1; i > 0; --i) {
		int target = int(drand48() * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
}

static int* perlin_generate_perm()
{
	int* p = new int[256];
	for (int i = 0; i < 256; ++i)
		p[i] = i;

	permute(p, 256);
	return p;
}

vec3* perlin::ranvec = perlin_generate();
int* perlin::perm_x = perlin_generate_perm();
int* perlin::perm_y = perlin_generate_perm();
int* perlin::perm_z = perlin_generate_perm();
