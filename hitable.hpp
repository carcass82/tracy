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
#include "material.hpp"
#include "texture.hpp"

class material;
class isotropic;

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


class xy_rect : public hitable
{
public:
	xy_rect() {}
	xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		float t = (k - r.origin().z) / r.direction().z;
		if (t < t_min || t > t_max)
			return false;

		float x = r.origin().x + t * r.direction().x;
		float y = r.origin().y + t * r.direction().y;
		if (x < x0 | x > x1 || y < y0 || y > y1)
			return false;

		rec.u = (x - x0) / (x1 - x0);
		rec.v = (y - y0) / (y1 - y0);
		rec.t = t;
		rec.mat_ptr = mp;
		rec.p = r.point_at_parameter(t);
		rec.normal = vec3(0, 0, 1);
		return true;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
		return true;
	}

	material* mp;
	float x0;
	float x1;
	float y0;
	float y1;
	float k;
};


class xz_rect : public hitable
{
public:
	xz_rect() {}
	xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat)
		: x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		float t = (k - r.origin().y) / r.direction().y;
		if (t < t_min || t > t_max)
			return false;

		float x = r.origin().x + t * r.direction().x;
		float z = r.origin().z + t * r.direction().z;
		if (x < x0 | x > x1 || z < z0 || z > z1)
			return false;

		rec.u = (x - x0) / (x1 - x0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		rec.mat_ptr = mp;
		rec.p = r.point_at_parameter(t);
		rec.normal = vec3(0, 0, 1);
		return true;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
		return true;
	}

	material* mp;
	float x0;
	float x1;
	float z0;
	float z1;
	float k;
};


class yz_rect : public hitable
{
public:
	yz_rect() {}
	yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat)
		: y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		float t = (k - r.origin().x) / r.direction().x;
		if (t < t_min || t > t_max)
			return false;

		float y = r.origin().y + t * r.direction().y;
		float z = r.origin().z + t * r.direction().z;
		if (y < y0 | y > y1 || z < z0 || z > z1)
			return false;

		rec.u = (y - y0) / (y1 - y0);
		rec.v = (z - z0) / (z1 - z0);
		rec.t = t;
		rec.mat_ptr = mp;
		rec.p = r.point_at_parameter(t);
		rec.normal = vec3(0, 0, 1);
		return true;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
		return true;
	}

	material* mp;
	float y0;
	float y1;
	float z0;
	float z1;
	float k;
};


class flip_normals : public hitable
{
public:
	flip_normals(hitable* p) : ptr(p) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		if (ptr->hit(r, t_min, t_max, rec)) {
			rec.normal = -rec.normal;
			return true;
		} else {
			return false;
		}
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		return ptr->bounding_box(t0, t1, box);
	}

	hitable* ptr;
};


class translate : public hitable
{
public:
	translate(hitable* p, const vec3& displacement) : ptr(p), offset(displacement) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		ray moved_r(r.origin() - offset, r.direction(), r.time());
		if (ptr->hit(moved_r, t_min, t_max, rec)) {
			rec.p += offset;
			return true;
		}
		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		if (ptr->bounding_box(t0, t1, box)) {
			box = aabb(box.min() + offset, box.max() + offset);
			return true;
		}
		return false;
	}

	hitable* ptr;
	vec3 offset;
};


class rotate_y : public hitable
{
public:
	rotate_y(hitable* p, float angle)
		:ptr(p)
	{
		float radians = glm::radians(angle);
		sin_theta = sin(radians);
		cos_theta = cos(radians);
		hasbox = ptr->bounding_box(0, 1, bbox);

		vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
		vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				for (int k = 0; k < 2; ++k) {
					float x = i * bbox.max().x + (1 - i) * bbox.min().x;
					float y = j * bbox.max().y + (1 - j) * bbox.min().y;
					float z = k * bbox.max().z + (1 - k) * bbox.min().z;
					float newx = cos_theta * x + sin_theta * z;
					float newz = -sin_theta * x + cos_theta * z;

					vec3 tester(newx, y, newz);
					for (int c = 0; c < 3; ++c) {
						if (tester[c] > max[c]) max[c] = tester[c];
						if (tester[c] < min[c]) min[c] = tester[c];
					}
				}
			}
		}

		bbox = aabb(min, max);
	}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		vec3 origin = r.origin();
		vec3 direction = r.direction();

		origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
		origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
		direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
		direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

		ray rotated_r(origin, direction, r.time());

		if (ptr->hit(rotated_r, t_min, t_max, rec)) {
			vec3 p = rec.p;
			vec3 normal = rec.normal;

			p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
			p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

			normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
			normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

			rec.p = p;
			rec.normal = normal;
			return true;
		}

		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = bbox;
		return hasbox;
	}

	hitable* ptr;
	bool hasbox;
	aabb bbox;
	float sin_theta;
	float cos_theta;
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



class box : public hitable
{
public:
	box() {}
	box(const vec3& p0, const vec3& p1, material* ptr)
		: pmin(p0), pmax(p1)
	{
		hitable** list = new hitable*[6];

		list[0] = new xy_rect(p0.x, p1.x, p0.y, p1.y, p1.z, ptr);
		list[1] = new flip_normals(new xy_rect(p0.x, p1.x, p0.y, p1.y, p0.z, ptr));
		list[2] = new xz_rect(p0.x, p1.x, p0.z, p1.z, p1.y, ptr);
		list[3] = new flip_normals(new xz_rect(p0.x, p1.x, p0.z, p1.z, p0.y, ptr));
		list[4] = new yz_rect(p0.y, p1.y, p0.z, p1.z, p1.x, ptr);
		list[5] = new flip_normals(new yz_rect(p0.y, p1.y, p0.z, p1.z, p0.x, ptr));

		list_ptr = new hitable_list(list, 6);
	}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		return list_ptr->hit(r, t_min, t_max, rec);
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		box = aabb(pmin, pmax);
		return true;
	}

	vec3 pmin, pmax;
	hitable* list_ptr;
};


class constant_medium : public hitable
{
public:
	constant_medium(hitable* b, float d, texture* a) : boundary(b), density(d)
	{
		phase_function = new isotropic(a);
	}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override
	{
		//bool db = (drand48() < 0.00001);
		bool db = false;

		hit_record rec1, rec2;
		if (boundary->hit(r, -FLT_MAX, FLT_MAX, rec1)) {
			if (boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2)) {
				if (rec1.t < t_min) rec1.t = t_min;
				if (rec2.t > t_max) rec2.t = t_max;
				if (rec1.t >= rec2.t) return false;
				if (rec1.t < 0) rec1.t = 0;

				float distance_inside_boundary = (rec2.t - rec1.t) * r.direction().length();
				float hit_distance = -(1 / density) * log(drand48());
				if (hit_distance < distance_inside_boundary) {
					rec.p = r.point_at_parameter(rec.t);
					rec.normal = vec3(1,0,0);
					rec.mat_ptr = phase_function;
					return true;
				}
			}
		}
		return false;
	}

	virtual bool bounding_box(float t0, float t1, aabb& box) const override
	{
		return boundary->bounding_box(t0, t1, box);
	}

	material* phase_function;
	float density;
	hitable* boundary;
};
