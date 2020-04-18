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

#include "common.h"
#include "ray.h"
#include "aabb.h"

namespace accel
{

struct Node
{
	Node() {}
	Node(const BBox& in_aabb) : aabb{ in_aabb } {}

	bool empty() const { return elem_start == elem_end; }

	size_t elem_start{ 0 };
	size_t elem_end{ 0 };
	Node* left{ nullptr };
	Node* right{ nullptr };
	BBox aabb{ FLT_MAX, -FLT_MAX };
};

template <typename T>
struct Tree
{
	Node* root;
	vector<T> elems;
};

template <typename T, int FIXED_SIZE>
class FixedSizeStack
{
public:
	void push(T item)   { array_[++head_] = item; }
	T pop()             { return array_[head_--]; }
	bool empty() const  { return head_ == -1; }
	bool full() const   { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	T array_[FIXED_SIZE];
};

template<typename T>
using ObjectAABBTesterFunction = std::function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectRayTesterFunction = std::function<bool(const T* elems, size_t count, const Ray&, HitData&)>;

template<typename T, size_t MIN_OBJECTS = 16, size_t MAX_DEPTH = 32>
Node* BuildTree(Tree<T>* tree, const vector<const T*>& objects, const BBox& box, ObjectAABBTesterFunction<T> ObjectBoxTester, size_t depth = 0)
{
	Node* node = new Node(box);

	if (objects.size() <= MIN_OBJECTS || depth >= MAX_DEPTH)
	{
		node->elem_start = tree->elems.size();
		for (const T* object : objects)
		{
			tree->elems.push_back(*object);
		}
		node->elem_end = tree->elems.size();

		return node;
	}
	

	float split_cost = FLT_MAX;
	int split_candidate = 5;
	int axis_candidate = depth % 3;

	vector<const T*> right;
	vector<const T*> left;

	//for (int axis = 0; axis < 3; ++axis)
	{
		// DEBUG: why is this faster than choosing best cost from all 3 axis?
		int axis = depth % 3;

		for (int i = 1; i < 10; ++i)
		{
			int right_count = 0;
			int left_count = 0;

			vec3 split_right = ((box.maxbound - box.minbound) / 10.f * (float)i) + EPS;
			vec3 split_left = ((box.maxbound - box.minbound) / 10.f * (float)(10 - i)) + EPS;

			BBox right_bbox{ box };
			right_bbox.maxbound[axis] -= split_left[axis];

			BBox left_bbox{ box };
			left_bbox.minbound[axis] += split_right[axis];

			for (const T* object : objects)
			{
				if (ObjectBoxTester(*object, right_bbox))
				{
					++right_count;
				}
				if (ObjectBoxTester(*object, left_bbox))
				{
					++left_count;
				}
			}

			float cost = (i / 10.f) * right_count + ((10 - i) / 10.f) * left_count;
			if (cost < split_cost)
			{
				axis_candidate = axis;
				split_candidate = i;
				split_cost = cost;
			}
		}
	}

	
	right.clear();
	left.clear();

	vec3 split_right = ((box.maxbound - box.minbound) / 10.f * (float)split_candidate) + EPS;
	vec3 split_left = ((box.maxbound - box.minbound) / 10.f * (float)(10 - split_candidate)) + EPS;

	BBox right_bbox{ box };
	right_bbox.maxbound[axis_candidate] -= split_left[axis_candidate];

	BBox left_bbox{ box };
	left_bbox.minbound[axis_candidate] += split_right[axis_candidate];

	for (const T* object : objects)
	{
		if (ObjectBoxTester(*object, right_bbox))
		{
			right.push_back(object);
		}
		if (ObjectBoxTester(*object, left_bbox))
		{
			left.push_back(object);
		}
	}

	node->left = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(tree, left, left_bbox, ObjectBoxTester, depth + 1);
	node->right = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(tree, right, right_bbox, ObjectBoxTester, depth + 1);

	return node;
}

template <typename T, size_t STACK_SIZE>
bool IntersectsWithTree(const Tree<T>* tree, const Ray& ray, HitData& inout_intersection, const ObjectRayTesterFunction<T>& ObjectTester)
{
	FixedSizeStack<const Node*, STACK_SIZE> to_be_tested;

	bool hit_something = false;
	
	const Node* current = tree->root;

	while (current || !to_be_tested.empty())
	{
		while (current && IntersectsWithBoundingBox(current->aabb, ray))
		{
			to_be_tested.push(current);
			current = current->left;
		}

		if (!to_be_tested.empty())
		{
			current = to_be_tested.pop();
			
			if (!current->empty() && ObjectTester(&tree->elems[current->elem_start], current->elem_end - current->elem_start, ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		current = current->right;
	}

	return hit_something;
}

}
