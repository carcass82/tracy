/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <functional>
using std::function;

#include <optional>
using std::optional;
using std::nullopt;

#include "common.h"
#include "ray.h"

#define USE_KDTREE 0

namespace accel
{

struct Triangle
{
	Triangle(const vec3& v0, const vec3& v1, const vec3& v2, int in_mesh_idx, int in_tri_idx)
		: vertices{ v0, v1, v2 }
		, v0v1{ v1 - v0 }
		, v0v2{ v2 - v0 }
		, mesh_idx{ in_mesh_idx }
		, tri_idx{ in_tri_idx }
	{}

	union {
		vec3 vertices[3];
		struct { vec3 v0; vec3 v1; vec3 v2; };
	};
	
	vec3 v0v1;
	vec3 v0v2;

	int mesh_idx;
	int tri_idx;
};

template <typename T>
struct Tree
{
	Tree()
		: children{ nullptr, nullptr }
	{}

	Tree(const BBox& in_aabb)
		: children{ nullptr, nullptr }
		, aabb{ in_aabb }
	{}

	BBox aabb;
	Tree<T>* children[2];
	vector<T> elems;
};

template<typename T>
using ObjectAABBTesterFunction = std::function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectRayTesterFunction = std::function<bool(const vector<T>, const Ray&, HitData&)>;

template<typename T, int MIN_OBJECTS, int MAX_DEPTH>
Tree<T>* BuildTree(const vector<const T*>& objects, const BBox& box, ObjectAABBTesterFunction<T> ObjectBoxTester, int depth = 0)
{
	Tree<T>* tree = new Tree<T>(box);

	if (objects.empty())
	{
		return tree;
	}

	if (objects.size() <= MIN_OBJECTS || depth >= MAX_DEPTH)
	{
		for (const T* object : objects)
		{
			tree->elems.push_back(*object);
		}

		return tree;
	}

	vector<const T*> forward;
	vector<const T*> backward;

	vec3 half_axis_size = ((box.maxbound - box.minbound) / 2.f) + 1.e-8f;
	int axis_selector = depth % 3;

	BBox backward_bbox{ box };
	backward_bbox.maxbound[axis_selector] -= half_axis_size[axis_selector];
	
	BBox forward_bbox{ box };
	forward_bbox.minbound[axis_selector] += half_axis_size[axis_selector];

	for (const T* object : objects)
	{
		if (ObjectBoxTester(*object, forward_bbox))
		{
			forward.push_back(object);
		}
		if (ObjectBoxTester(*object, backward_bbox))
		{
			backward.push_back(object);
		}
	}

	tree->children[0] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(forward, forward_bbox, ObjectBoxTester, depth + 1);
	tree->children[1] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(backward, backward_bbox, ObjectBoxTester, depth + 1);

	return tree;
}

template <typename T>
bool IntersectsWithTree(const Tree<T>& tree, const Ray& ray, HitData& inout_intersection, ObjectRayTesterFunction<T> ObjectTester)
{
	bool hit_something = false;
	
	const Tree<T>* root = &tree;
	if (root && IntersectsWithBoundingBox(root->aabb, ray))
	{
		hit_something = (root->children[0] && IntersectsWithTree(*root->children[0], ray, inout_intersection, ObjectTester)) ||
	                    (root->children[1] && IntersectsWithTree(*root->children[1], ray, inout_intersection, ObjectTester)) ||
	                    ObjectTester(root->elems, ray, inout_intersection);
	}
	
	return hit_something;
}

}
