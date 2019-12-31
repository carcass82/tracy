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

struct Node
{
	Node() {}
	Node(const BBox& in_aabb) : aabb{ in_aabb } {}

	bool empty() const { return elem_start == elem_end; }

	size_t elem_start{ 0 };
	size_t elem_end{ 0 };
	Node* children[2]{ nullptr, nullptr };
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
	void push(T item)   { array_[++head_] = item; if (head_ >= FIXED_SIZE) DEBUG_BREAK(); }
	T pop()             { return array_[head_--]; if (head_ < -1) DEBUG_BREAK(); }
	bool empty() const  { return head_ == -1; }
	bool full() const   { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	T array_[FIXED_SIZE];
};

template<typename T>
using ObjectAABBTesterFunction = std::function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectRayTesterFunction = std::function<bool(const T* start, const T* end, const Ray&, HitData&)>;

template<typename T, size_t MIN_OBJECTS, size_t MAX_DEPTH>
Node* BuildTree(Tree<T>* tree, const vector<const T*>& objects, const BBox& box, ObjectAABBTesterFunction<T> ObjectBoxTester, int depth = 0)
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

	node->children[0] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(tree, forward, forward_bbox, ObjectBoxTester, depth + 1);

	node->children[1] = BuildTree<T, MIN_OBJECTS, MAX_DEPTH>(tree, backward, backward_bbox, ObjectBoxTester, depth + 1);
	
	return node;
}

template <typename T, size_t STACK_SIZE>
bool IntersectsWithTree(const Tree<T>* tree, const Ray& ray, HitData& inout_intersection, ObjectRayTesterFunction<T> ObjectTester)
{
	FixedSizeStack<const Node*, STACK_SIZE> to_be_tested;

	bool hit_something = false;
	
	const Node* root = tree->root;

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
			
			if (!root->empty() && ObjectTester(&tree->elems[root->elem_start], &tree->elems[root->elem_end] + 1, ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		root = root->children[1];
	}

	return hit_something;
}

}
