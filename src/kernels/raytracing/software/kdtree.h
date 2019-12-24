/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
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
		, mesh_idx(in_mesh_idx)
		, tri_idx(in_tri_idx)
	{}

	vec3 GetCenter() const
	{
		return (vertices[0] + vertices[1] + vertices[2]) / 3.f;
	}

	vec3 GetNormal() const
	{
		return normalize(cross(v0v1, v0v2));
	}

	vec3 vertices[3];
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

	BBox aabb;
	Tree<T>* children[2];
	vector<T> elems;
};

bool IsInsideAABB(const Triangle& triangle, const BBox& aabb)
{
	for (int i = 0; i < 3; ++i)
	{
		if (!(triangle.vertices[i].x <= aabb.maxbound.x && triangle.vertices[i].x >= aabb.minbound.x) &&
			 (triangle.vertices[i].y <= aabb.maxbound.y && triangle.vertices[i].y >= aabb.minbound.y) &&
			 (triangle.vertices[i].z <= aabb.maxbound.z && triangle.vertices[i].z >= aabb.minbound.z))
		{
			return false;
		}
	}

	return true;
}

template<typename T, int MIN_OBJECTS, int MAX_DEPTH>
Tree<T>* BuildTree(const vector<const T*>& objects, BBox box, int depth = 0)
{
	Tree<T>* tree = new Tree<T>;
	tree->aabb = box;
	tree->children[0] = nullptr;
	tree->children[1] = nullptr;

	if (objects.empty())
	{
		return tree;
	}

	if (objects.size() <= MIN_OBJECTS || depth > MAX_DEPTH)
	{
		for (auto&& object : objects)
		{
			tree->elems.push_back(*object);
		}
		return tree;
	}

	vector<const T*> forward;
	vector<const T*> backward;

	vec3 half_axis_size = ((box.maxbound - box.minbound) / 2.f) - 1.e-8f;
	int axis_selector = depth % 3;

	BBox backward_bbox{ box };
	backward_bbox.maxbound[axis_selector] -= half_axis_size[axis_selector];
	
	BBox forward_bbox{ box };
	forward_bbox.minbound[axis_selector] += half_axis_size[axis_selector];

	for (auto&& object : objects)
	{
		if (IsInsideAABB(*object, forward_bbox))
		{
			forward.push_back(object);
		}
		if (IsInsideAABB(*object, backward_bbox))
		{
			backward.push_back(object);
		}
	}

	tree->children[0] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(forward, forward_bbox, depth + 1);
	tree->children[1] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(backward, backward_bbox, depth + 1);

	return tree;
}

template <typename T, class Predicate>
vector<T> IntersectsWithTree(const Tree<T>& tree, const Ray& ray, Predicate BoxTester)
{
	vector<T> EMPTY;

	const Tree<T>* root = &tree;
	if (BoxTester(root->aabb, ray))
	{
		while (root->children[0] || root->children[1])
		{
			if (root->children[0] && BoxTester(root->children[0]->aabb, ray))
			{
				root = root->children[0];
			}
			else if (root->children[1] /* && BoxTester(root->children[1]->aabb, ray) */)
			{
				root = root->children[1];
			}
			else
			{
				DEBUG_BREAK();
			}
		}
		return root->elems;
	}
	else
	{
		return EMPTY;
	}
}
}
