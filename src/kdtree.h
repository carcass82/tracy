/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <vector>

#include "common.h"
#include "ray.h"
#include "aabb.h"

namespace accel
{

template <typename T, int FIXED_SIZE>
class FixedSizeStack
{
public:
	CUDA_CALL void push(T item) { array_[++head_] = item; }
	CUDA_CALL T pop() { return array_[head_--]; }
	CUDA_CALL bool empty() const { return head_ == -1; }
	CUDA_CALL bool full() const { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	T array_[FIXED_SIZE];
};

struct Node
{
	CUDA_DEVICE_CALL Node() {}
	CUDA_DEVICE_CALL Node(const BBox& in_aabb, unsigned int in_depth = 0) : aabb(in_aabb), depth(in_depth) {}

	CUDA_DEVICE_CALL bool empty() const { return elem_start == elem_end; }

	unsigned int elem_start{ 0 };
	unsigned int elem_end{ 0 };
	Node* left{ nullptr };
	Node* right{ nullptr };
	BBox aabb{ FLT_MAX, -FLT_MAX };
	unsigned int depth;
};

template <typename T, template<class...> class Container = std::vector>
struct Tree
{
	Node* root;
	Container<T> elems;
};

template <typename OptimizedT>
struct OptimizedTree
{
	Node* root;
	OptimizedT elems;
};

template<typename T>
using ObjectAABBTesterFunction = function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectRayTesterFunction = function<bool(const T* elems, unsigned start, unsigned count, const Ray&, HitData&)>;

struct SplitInfo
{
	int axis_candidate = axis_candidate;
	float split_cost = FLT_MAX;
	unsigned int right_count = 0;
	unsigned int left_count = 0;
	BBox right_box;
	BBox left_box;
};

template<typename T, template<class...> class Container = std::vector>
CUDA_DEVICE_CALL inline void GetBestSplit(SplitInfo& result, const BBox& current_bbox, unsigned int current_depth, const Container<T>& objects, const ObjectAABBTesterFunction<T>& ObjectBoxTester)
{
#if SAH_NOT_WORKING
	for (int axis = 0; axis < 3; ++axis)
	{
		for (int i = 1; i < 10; ++i)
		{
			unsigned int right_count = 0;
			unsigned int left_count = 0;

			float split_max = (current_bbox.GetSize()[axis] / 10.f * i) + 1.e-4f;
			float split_min = (current_bbox.GetSize()[axis] / 10.f * (10 - i)) - 1.e-4f;

			BBox left_bbox(current_bbox);
			left_bbox.maxbound[axis] -= split_min;

			BBox right_bbox(current_bbox);
			right_bbox.minbound[axis] += split_max;

			for (const T& object : objects)
			{
				if (ObjectBoxTester(object, right_bbox))
				{
					++right_count;
				}
				if (ObjectBoxTester(object, left_bbox))
				{
					++left_count;
				}
			}

			float cost = (i / 10.f) * right_count + ((10 - i) / 10.f) * left_count;
			if (cost < result.split_cost)
			{
				result.split_cost = cost;

				result.axis_candidate = axis;
				
				result.left_box = left_bbox;
				result.left_count = left_count;

				result.right_box = right_bbox;
				result.right_count = right_count;
			}
		}
	}
#else
	unsigned int axis = current_depth % 3;
	float split = (current_bbox.GetSize()[axis] / 2.f) + 1.e-4f;

	BBox left_bbox(current_bbox);
	left_bbox.maxbound[axis] -= split;

	BBox right_bbox(current_bbox);
	right_bbox.minbound[axis] += split;

	unsigned int right_count = 0;
	unsigned int left_count = 0;
	for (const T& object : objects)
	{
		if (ObjectBoxTester(object, right_bbox))
		{
			++right_count;
		}
		if (ObjectBoxTester(object, left_bbox))
		{
			++left_count;
		}
	}

	result.axis_candidate = axis;
	
	result.left_box = left_bbox;
	result.left_count = left_count;

	result.right_box = right_bbox;
	result.right_count = right_count;
#endif
}

template<typename T, template<class...> class Container = std::vector, class TreeType = Tree<T, Container>>
CUDA_DEVICE_CALL inline void CopyObjectsToNode(TreeType* tree, Node* current_node, const Container<T>& objects, const BBox& current_box, const ObjectAABBTesterFunction<T>& ObjectBoxTester)
{
	current_node->elem_start = (unsigned)tree->elems.size();
	for (const T& object : objects)
	{
		if (ObjectBoxTester(object, current_box))
		{
			tree->elems.push_back(object);
		}
	}
	current_node->elem_end = (unsigned)tree->elems.size();
}

template<typename T,
         template<class...> class Container = std::vector,
         class TreeType = Tree<T, Container>,
         unsigned MIN_OBJECTS = 16,
         unsigned MAX_DEPTH = 32>
CUDA_DEVICE_CALL inline void BuildTree(TreeType* tree,                                     // output tree
                                       const Container<T>& objects,                        // temp objects used while filling the tree
                                       const BBox& box,                                    // in: initial bbox, temp used while filling the tree
                                       const ObjectAABBTesterFunction<T>& ObjectBoxTester) // tester used for T objects agains BBox while filling the tree
{
	tree->root = new Node(box);

	Container<Node*> build_queue;
	build_queue.push_back(tree->root);

	while(!build_queue.empty())
	{
		Node* current_node = build_queue.back();
		build_queue.pop_back();

		BBox current_bbox = current_node->aabb;
		unsigned int current_depth = current_node->depth;

		SplitInfo split_helper;
		GetBestSplit(split_helper, current_bbox, current_depth, objects, ObjectBoxTester);

		if (split_helper.right_count > 0)
		{
			current_node->right = new Node(split_helper.right_box, current_depth + 1);
			if (split_helper.right_count <= MIN_OBJECTS || current_depth > MAX_DEPTH)
			{
				CopyObjectsToNode(tree, current_node->right, objects, split_helper.right_box, ObjectBoxTester);
			}
			else
			{
				build_queue.push_back(current_node->right);
			}
		}

		if (split_helper.left_count > 0)
		{
			current_node->left = new Node(split_helper.left_box, current_depth + 1);
			if (split_helper.left_count <= MIN_OBJECTS || current_depth > MAX_DEPTH)
			{
				CopyObjectsToNode(tree, current_node->left, objects, split_helper.left_box, ObjectBoxTester);
			}
			else
			{
				build_queue.push_back(current_node->left);
			}
		}
	}
}

template <typename TreeType, typename ElemType, size_t STACK_SIZE = 32 + 1>
CUDA_DEVICE_CALL
bool IntersectsWithTree(const TreeType* tree, const Ray& ray, HitData& inout_intersection, const ObjectRayTesterFunction<ElemType>& ObjectTester)
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

			if (!current->empty() && ObjectTester(&tree->elems[0], current->elem_start, current->elem_end, ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		current = current->right;
	}

	return hit_something;
}

}
