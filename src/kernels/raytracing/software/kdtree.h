/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include "common.h"
#include "ray.h"

namespace accel
{
bool IntersectsWithAABB(const BBox& a, const BBox& b)
{
	return (!((a.maxbound.x < b.minbound.x || a.minbound.x > b.maxbound.x) ||
	          (a.maxbound.y < b.minbound.y || a.minbound.y > b.maxbound.y) ||
	          (a.maxbound.z < b.minbound.z || a.minbound.z > b.maxbound.z)));
}

template <typename T>
struct Tree
{
	Tree()
		: children{}
	{}

	BBox aabb;
	Tree<T>* children[2];
	vector<T> elems;
};

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
		if (IntersectsWithAABB(object->GetAABB(), forward_bbox))
		{
			forward.push_back(object);
		}
		else if (IntersectsWithAABB(object->GetAABB(), backward_bbox))
		{
			backward.push_back(object);
		}
	}

	tree->children[0] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(forward, forward_bbox, depth + 1);
	tree->children[1] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(backward, backward_bbox, depth + 1);

	return tree;
}

template <typename T>
vector<T> IntersectsWithTree(const Tree<T>& tree, const Ray& ray)
{
	vector<T> EMPTY;

	const Tree<T>* root = &tree;
	if (IntersectsWithBoundingBox(root->aabb, ray))
	{
		while (root->children[0] || root->children[1])
		{
			if (root->children[0] && IntersectsWithBoundingBox(root->children[0]->aabb, ray))
			{
				root = root->children[0];
			}
			else if (root->children[1] /* && IntersectsWithBoundingBox(root->children[1]->aabb, ray) */)
			{
				root = root->children[1];
			}
			else
			{
				break;
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
