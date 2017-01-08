/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibook
 *
 * (c) Carlo Casta, 2017
 */
#pragma once

#include <iostream>

#include "glm/glm.hpp"

#include "geom.hpp"
#include "ray.hpp"
#include "aabb.hpp"

class material;

struct hit_record
{
	float t;
	float u;
	float v;
	vec3 p;
	vec3 normal;
	material* mat_ptr;
};

class hitable
{
public:
	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};

class sphere : public hitable
{
public:
	sphere()                              {}
	sphere(vec3 c, float r, material* m)
		: center(c), radius(r), mat(m)    {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;

		if (discriminant > 0) {

			float temp = (-b - sqrtf(b * b - a * c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(temp);
				rec.normal = (rec.p - center) / radius;
				get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
				rec.mat_ptr = mat;

				return true;
			}

			temp = (-b + sqrtf(b * b - a * c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(temp);
				rec.normal = (rec.p - center) / radius;
				get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
				rec.mat_ptr = mat;

				return true;
			}

		}

		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(center - vec3(radius), center + vec3(radius));
		return true;
	}

	vec3 center;
	float radius;
	material* mat;
};

class moving_sphere : public hitable
{
public:
	moving_sphere()                                                                   {}
	moving_sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m)
		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat(m)       {}

	vec3 center(float time) const { return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0); }

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		vec3 oc = r.origin() - center(r.time());
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;

		if (discriminant > 0) {

			float temp = (-b - sqrtf(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center(r.time())) / radius;
				get_sphere_uv((rec.p - center(r.time())) / radius, rec.u, rec.v);
				rec.mat_ptr = mat;

				return true;
			}

			temp = (-b + sqrtf(discriminant)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center(r.time())) / radius;
				get_sphere_uv((rec.p - center(r.time())) / radius, rec.u, rec.v);
				rec.mat_ptr = mat;

				return true;
			}

		}

		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(glm::min(center(t0) - vec3(radius), center(t1) - vec3(radius)), glm::max(center(t0) + vec3(radius), center(t1) + vec3(radius)));
		return true;
	}

	vec3 center0;
	vec3 center1;
	float time0;
	float time1;
	float radius;
	material* mat;	
};

int box_x_compare(const void* a, const void* b)
{
	aabb box_left, box_right;
	hitable* ah = *(hitable**)a;
	hitable* bh = *(hitable**)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		std::cerr << "no bbox in bvh_node constructor!\n";

	return (box_left.min().x - box_right.min().x < 0.0f)? -1 : 1;
}

int box_y_compare(const void* a, const void* b)
{
	aabb box_left, box_right;
	hitable* ah = *(hitable**)a;
	hitable* bh = *(hitable**)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		std::cerr << "no bbox in bvh_node constructor!\n";

	return (box_left.min().y - box_right.min().y < 0.0f)? -1 : 1;
}

int box_z_compare(const void* a, const void* b)
{
	aabb box_left, box_right;
	hitable* ah = *(hitable**)a;
	hitable* bh = *(hitable**)b;

	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		std::cerr << "no bbox in bvh_node constructor!\n";

	return (box_left.min().z - box_right.min().z < 0.0f)? -1 : 1;
}

class bvh_node : public hitable
{
public:
	bvh_node() {}
	bvh_node(hitable **l, int n, float time0, float time1)
	{
		int axis = int(3 * drand48());
		if (axis == 0)
			qsort(l, n, sizeof(hitable*), box_x_compare);
		else if (axis == 1)
			qsort(l, n, sizeof(hitable*), box_y_compare);
		else //axis == 2
			qsort(l, n, sizeof(hitable*), box_z_compare);

		if (n == 1) {
			left = right = l[0];
		} else if (n == 2) {
			left = l[0];
			right = l[1];
		} else {
			left = new bvh_node(l, n / 2, time0, time1);
			right = new bvh_node(l + n / 2, n - n / 2, time0, time1);
		}

		aabb box_left, box_right;
		if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
			std::cerr << "no bbox in bvh_node constructor!\n";

		box = surrounding_box(box_left, box_right);
	}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		if (box.hit(r, t_min, t_max)) {
			hit_record left_rec;
			bool hit_left = left->hit(r, t_min, t_max, left_rec);

			hit_record right_rec;
			bool hit_right = right->hit(r, t_min, t_max, right_rec);

			if (hit_left && hit_right) {
				if (left_rec.t < right_rec.t)
					rec = left_rec;
				else
					rec = right_rec;
				return true;
			} else if (hit_left) {
				rec = left_rec;
				return true;
			} else if (hit_right) {
				rec = right_rec;
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

	virtual bool bounding_box(float t0, float t1, aabb& b) const override
	{
		b = box;
		return true;
	}

	hitable* left;
	hitable* right;
	aabb box;
};


class hitable_list : public hitable
{
public:
	hitable_list() {}
	hitable_list(hitable** l, int n) : list(l), list_size(n) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		hit_record temp_rec;
		bool hit_anything = false;

		double closest_so_far = t_max;
		for (int i = 0; i < list_size; ++i) {
			if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		if (list_size < 1)
			return false;

		aabb temp_box;
		bool first_true = list[0]->bounding_box(t0, t1, temp_box);
		if (!first_true)
			return false;
		else
			box = temp_box;

		for (int i = 1; i < list_size; ++i) {
			if (list[0]->bounding_box(t0, t1, temp_box)) {
				box = surrounding_box(box, temp_box);
			} else {
				return false;
			}
		}
		return true;
	}

	hitable** list;
	int list_size;
};
