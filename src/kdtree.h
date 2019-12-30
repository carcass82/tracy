/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <vector>
using std::vector;

#include <functional>
using std::function;

#include <optional>
using std::optional;
using std::nullopt;

#include "common.h"
#include "ray.h"
#include "aabb.h"

namespace accel
{

template <typename T>
struct Tree
{
	Tree()
		: children{ nullptr, nullptr }
	{}

	Tree(const BBox& in_aabb)
		: aabb{ in_aabb }
		, children{ nullptr, nullptr }
		
	{}

	BBox aabb;
	Tree<T>* children[2];
	vector<T> elems;
};

template <typename T, int FIXED_SIZE>
class StaticStack
{
public:
	void push(const T* item) { array_[++head_] = item; }
	const T* pop()           { return array_[head_--]; }
	bool empty() const       { return head_ == -1; }
	bool full() const        { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	const T* array_[FIXED_SIZE];
};

template<typename T>
using ObjectAABBTesterFunction = std::function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectRayTesterFunction = std::function<bool(const vector<T>&, const Ray&, HitData&)>;

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

template <typename T, int STACK_SIZE>
bool IntersectsWithTree(const Tree<T>* tree, const Ray& ray, HitData& inout_intersection, ObjectRayTesterFunction<T> ObjectTester)
{
	StaticStack<Tree<T>, STACK_SIZE> to_be_tested;

	bool hit_something = false;
	const Tree<T>* root = tree;
	while (root || !to_be_tested.empty())
	{
		while (root)
		{
			if (IntersectsWithBoundingBox(root->aabb, ray))
			{
				to_be_tested.push(root);
				root = root->children[0];
			}
			else
			{
				break;
			}
		}

		if (!to_be_tested.empty())
		{
			root = to_be_tested.pop();
			
			if (root->elems.size() > 0 && ObjectTester(root->elems, ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		root = root->children[1];
	}

	return hit_something;
}

}
