/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#include <iostream>
using std::cout;

#include <cfloat>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
using glm::vec3;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "noise.hpp"
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
		vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
			return emitted + attenuation * color(scattered, world, depth + 1);
		} else {
			return emitted;
		}

	} else {

		//vec3 unit_direction = glm::normalize(r.direction());
		//float t = 0.5f * (unit_direction.y + 1.0f);
		//return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
		
		return vec3();

		// debug - white "ambient" light
		//return vec3(1,1,1);
	}
}

hitable* random_scene()
{
	const int n = 500;
	hitable** list = new hitable*[n + 1];

	texture* terrain_texture = new checker_texture(new constant_texture(vec3(0.2, 0.3, 0.1)), new constant_texture(vec3(0.9, 0.9, 0.9)));
	list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(terrain_texture));

	int i = 1;
	for (int a = -10; a < 10; ++a) {
		for (int b = -10; b < 10; ++b) {
			float choose_mat = drand48();
			vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					//list[i++] = new moving_sphere(center, center + vec3(0, 0.5 * drand48(), 0.0), 0.0, 1.0, 0.2, new lambertian(new constant_texture(vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48())));
					list[i++] = new sphere(center, 0.2, new lambertian(new constant_texture(vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48()))));
				} else if (choose_mat < 0.95) {
					list[i++] = new sphere(center, 0.2, new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), 0.5 * drand48()));
				} else {
					list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}

			}
		}
	}

	// area light
	//list[i++] = new xy_rect(-2, 6, 0, 3, -3, new diffuse_light(new constant_texture(vec3(4,4,4))));
	list[i++] = new xz_rect(-20, 20, -20, 20, 10, new diffuse_light(new constant_texture(vec3(0.85,0.85,0.85))));

	// lambertian
	list[i++] = new sphere(vec3(-2, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));

	// dielectric
	list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));

	// metal
	list[i++] = new sphere(vec3(2, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0f));

	// lambertian noise ("marble like")
	list[i++] = new sphere(vec3(4, 1, 0), 1.0, new lambertian(new noise_texture(5.0f)));

	// lambertian textured
	int nx, ny, nn;
	unsigned char* tex_data = stbi_load("earth.jpg", &nx, &ny, &nn, 0);
	list[i++] = new sphere(vec3(6, 1, 0), 1.0, new lambertian(new image_texture(tex_data, nx, ny)));

	return new hitable_list(list, i);
}

hitable* cornellbox_scene()
{
	const int n = 50;
	hitable** list = new hitable*[n + 1];

	material* red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material* white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material* green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material* light = new diffuse_light(new constant_texture(vec3(1, 1, 1)));

	int i = 0;
	list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
	list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	list[i++] = new xz_rect(113, 443, 127, 432, 550, light);
	list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

	//list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130, 0, 65));
	list[i++] = new sphere(vec3(200, 100, 150), 100.0, new dielectric(1.5));
	list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white), 15), vec3(265, 0, 295));

	return new hitable_list(list, i);
}

hitable* final()
{
	int nb = 20;
	hitable** list = new hitable*[30];
	hitable** boxlist = new hitable*[10000];
	hitable** boxlist2 = new hitable*[10000];

	material* white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material* ground = new lambertian(new constant_texture(vec3(0.48, 0.83, 0.53)));
	material* light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));

	int b = 0;
	for (int i = 0; i < nb; ++i) {
		for (int j = 0; j < nb; ++j) {
			float w = 100;
			float x0 = -1000 + i * w;
			float z0 = -1000 + j * w;
			float y0 = 0;
			float x1 = x0 + w;
			float y1 = 100 * (drand48() + 0.01);
			float z1 = z0 + w;

			boxlist[b++] = new box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground);
		}
	}

	material* _lambertian = new lambertian(new constant_texture(vec3(0.7, 0.3, 0.1)));
	material* _dielectric = new dielectric(1.5);
	material* _metal = new metal(vec3(0.8, 0.8, 0.9), 10.0);

	int nx, ny, nn;
	unsigned char* tex_data = stbi_load("earth.jpg", &nx, &ny, &nn, 0);
	material* _textured = new lambertian(new image_texture(tex_data, nx, ny));

	material* _noise = new lambertian(new noise_texture(0.1));


	int l = 0;
	list[l++] = new bvh_node(boxlist, b, 0, 1);
	list[l++] = new xz_rect(123, 423, 147, 412, 554, light);
	list[l++] = new moving_sphere(vec3(400,400,200), vec3(430,400,200), 0, 1, 50, _lambertian);
	list[l++] = new sphere(vec3(260, 150, 45), 50, _dielectric);
	list[l++] = new sphere(vec3(0, 150, 145), 50, _metal);

	hitable* boundary = new sphere(vec3(360, 150, 145), 70, _dielectric);
	list[l++] = boundary;
	list[l++] = new constant_medium(boundary, 0.2, new constant_texture(vec3(0.2, 0.4, 0.9)));

	boundary = new sphere(vec3(0, 0, 0), 5000, _dielectric);
	list[l++] = new constant_medium(boundary, 0.0001, new constant_texture(vec3(1.0, 1.0, 1.0)));

	list[l++] = new sphere(vec3(400, 200, 400), 100, _textured);
	list[l++] = new sphere(vec3(220, 280, 300), 80, _noise);

	int ns = 1000;
	for (int i = 0; i < ns; ++i) {
		boxlist2[i] = new sphere(vec3(165 * drand48(), 165 * drand48(), 165 * drand48()), 10, white);
	}

	list[l++] = new translate(new rotate_y(new bvh_node(boxlist2, ns, 0.0, 1.0), 15), vec3(-100, 270, 395));

	return new hitable_list(list, l);
}

int main()
{
	const int nx = 500; // w
	const int ny = 500; // h
	const int ns = 1000; // samples

#if 0
	hitable *world = random_scene(); 

	// camera
	const vec3 look_from(10.0f, 1.5f, 4.0f);
	const vec3 look_at(2.0f, 0.5f, -2.0f);
	const float fov = 45.0f;
	const float dist_to_focus = length(look_from - look_at);
	const float aperture = 2.0f;
#endif

	hitable *world = cornellbox_scene();

#if 0
	hitable *world = final();
#endif

	// camera
	const vec3 look_from(278, 278, -800);
	const vec3 look_at(278, 278, 0);
	const float fov = 40.0f;
	const float dist_to_focus = 10;
	const float aperture = 0.0f;


	camera cam(look_from, look_at, vec3(0.0f, 1.0f, 0.0f), fov, float(nx) / float(ny), aperture, dist_to_focus, 0.0f, 1.0f);

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

			// ToneMap
			// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
			{
				col = (col * (2.51f * col + 0.03f)) / (col * (2.43f * col + 0.59f) + 0.14f);
			}

			// gamma correct 2.0
			col = glm::sqrt(col);

			cout << int(255.99 * col[0]) << " "
			     << int(255.99 * col[1]) << " "
			     << int(255.99 * col[2]) << "\n";
		}

	}
}