/*
 * simple raytracer
 *
 */
#include <iostream>
using std::cout;

#include <cfloat>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
using glm::vec3;

#include "material.hpp"
#include "camera.hpp"
#include "hitable.hpp"
#include "ray.hpp"
#include "geom.hpp"

vec3 color(const ray& r, hitable* world, int depth)
{
	hit_record rec;
	if (world->hit(r, 0.001f, FLT_MAX, rec)) {

		ray scattered;
		vec3 attenuation;

		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
			return attenuation * color(scattered, world, depth + 1);
		} else {
			return vec3(0.0f, 0.0f, 0.0f);
		}

	} else {

		vec3 unit_direction = glm::normalize(r.direction());
		float t = 0.5f * (unit_direction.y + 1.0f);
		return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);

	}
}

hitable* random_scene()
{
	const int n = 500;
	hitable** list = new hitable*[n + 1];
	list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));

	int i = 1;
	for (int a = -11; a < 11; ++a) {
		for (int b = -11; b < 11; ++b) {
			float choose_mat = drand48();
			vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48())));
				} else if (choose_mat < 0.95) {
					list[i++] = new sphere(center, 0.2, new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), 0.5 * drand48()));
				} else {
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}

			}
		}
	}

	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0f));

	return new hitable_list(list, i);
}

int main()
{
	const int nx = 1024; // w
	const int ny = 768; // h
	const int ns = 10; // samples

	hitable *world = random_scene(); 

	// camera
	const vec3 look_from(12.0f, 1.5f, 3.0f);
	const vec3 look_at(0.0f, 0.5f, -1.0f);
	const float fov = 45.0f;
	const float dist_to_focus = length(look_from - look_at);
	const float aperture = 2.0f;

	camera cam(look_from, look_at, vec3(0.0f, 1.0f, 0.0f), fov, float(nx) / float(ny), aperture, dist_to_focus);

	// PPM file header
	cout << "P3\n" << nx << " " << ny << "\n255\n";

	// path tracing
	for (int j = ny - 1; j >= 0; --j) {

		for (int i = 0; i < nx; ++i) {

			vec3 col(0, 0, 0);

#if 1
			#pragma omp parallel for
			for (int s = 0; s < ns; ++s) {

				float u = float(i + drand48()) / float(nx);
				float v = float(j + drand48()) / float(ny);

				ray r = cam.get_ray(u, v);

				vec3 temp_col = color(r, world, 0);

				#pragma omp critical
				col += temp_col;
			}
			col /= float(ns);
#else
			float u = float(i) / float(nx);
			float v = float(j) / float(ny);

			ray r = cam.get_ray(u, v);
			col = color(r, world, 0);
#endif

			// gamma correct 2.0
			col = glm::sqrt(col);

			cout << int(255.99 * col[0]) << " "
			     << int(255.99 * col[1]) << " "
			     << int(255.99 * col[2]) << "\n";
		}

	}
}