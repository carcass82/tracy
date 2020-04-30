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
constexpr unsigned int TREE_MAXDEPTH = 32;                 // all nodes with a depth of 32 become a leaf
constexpr unsigned int TREE_MINOBJECTS = 16;               // if after a split a node contains 16 objects or less it becomes a leaf
constexpr unsigned int TREE_USELESS_SPLIT_THRESHOLD = 200; // if a split produces 200% of initial nodes we say it's useless and nodes become a leaf


template <typename T, int FIXED_SIZE>
class FixedSizeStack
{
public:
	CUDA_CALL void Push(T item)    { array_[++head_] = item; }
	CUDA_CALL T    Pop()           { return array_[head_--]; }
	CUDA_CALL bool IsEmpty() const { return head_ == -1; }
	CUDA_CALL bool IsFull() const  { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	T array_[FIXED_SIZE];
};

template <typename T, template<class...> class Container = std::vector>
struct Node
{
	CUDA_DEVICE_CALL Node(unsigned int in_depth = 0)
		: depth(in_depth)
	{}

	CUDA_DEVICE_CALL Node(const BBox& in_aabb, unsigned int in_depth = 0)
		: aabb(in_aabb)
		, depth(in_depth)
	{}

	CUDA_DEVICE_CALL bool IsEmpty() const          { return elems.empty();              }
	CUDA_DEVICE_CALL unsigned int GetSize() const  { return (unsigned int)elems.size(); }

	Node* left = {};
	Node* right = {};
	BBox aabb = { FLT_MAX, -FLT_MAX };
	Container<T> elems;
	unsigned int depth;
};

template<typename T>
using ObjectAABBTesterFunction = function<bool(const T&, const BBox&)>;

template<typename T>
using ObjectsRayTesterFunction = function<bool(const T* elems, unsigned int count, const Ray&, HitData&)>;

template<typename T, template<class...> class Container = std::vector, class Predicate>
CUDA_DEVICE_CALL void CopyIf(const Container<T>& src, Container<T>& dest, Predicate Pred)
{
	for (auto& src_object : src)
	{
		if (Pred(src_object))
		{
			dest.push_back(src_object);
		}
	}
}

template<typename T, class NodeType>
CUDA_DEVICE_CALL inline unsigned int SplitAndGetDuplicationPercentage(const NodeType& current_node, NodeType& right_node, NodeType& left_node, const ObjectAABBTesterFunction<T>& ObjectBoxTester)
{
	constexpr float ROUND = 1.e-4f;

#if !defined(DISABLE_SAH)

	float best_cost = FLT_MAX;
	unsigned int best_axis = current_node.depth % 3;
	unsigned int best_split = 5;

	//for (int axis = 0; axis < 3; ++axis)
	int axis = best_axis;
	{
		for (int i = 1; i < 10; ++i)
		{
			unsigned int right_count = 0;
			unsigned int left_count = 0;

			vec3 split_unit = current_node.aabb.GetSize() / 10.f;

			BBox left_bbox(current_node.aabb);
			left_bbox.maxbound[axis] -= ((split_unit[axis] * i) + ROUND);

			BBox right_bbox(current_node.aabb);
			right_bbox.minbound[axis] += (split_unit[axis] * (10 - i)) + ROUND;

			for (const T& object : current_node.elems)
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

			float cost = (i / 10.f) * (2 * right_count) + ((10 - i) / 10.f) * (2 * left_count);
			if (cost < best_cost)
			{
				best_cost = cost;
				best_axis = axis;
				best_split = i;
			}
		}
	}

	float split_max = (current_node.aabb.GetSize()[best_axis] / 10.f * best_split) + ROUND;
	float split_min = (current_node.aabb.GetSize()[best_axis] / 10.f * (10 - best_split)) + ROUND;

	BBox left_bbox(current_node.aabb);
	left_bbox.maxbound[best_axis] -= split_min;
	left_node.aabb = left_bbox;

	BBox right_bbox(current_node.aabb);
	right_bbox.minbound[best_axis] += split_max;
	right_node.aabb = right_bbox;

#else

	unsigned int axis = current_node.depth % 3;
	float split = (current_node.aabb.GetSize()[axis] / 2.f) + ROUND;

	BBox left_bbox(current_node.aabb);
	left_bbox.maxbound[axis] -= split;
	left_node.aabb = left_bbox;

	BBox right_bbox(current_node.aabb);
	right_bbox.minbound[axis] += split;
	right_node.aabb = right_bbox;

#endif

	CopyIf(current_node.elems, right_node.elems, [&](const T& elem) { return ObjectBoxTester(elem, right_node.aabb); });
	CopyIf(current_node.elems, left_node.elems,  [&](const T& elem) { return ObjectBoxTester(elem, left_node.aabb); });

	return (right_node.GetSize() + left_node.GetSize()) * 100 / current_node.GetSize();
}


template<typename ElemType,
         template<class...> class Container = std::vector,
         unsigned int MIN_OBJECTS = TREE_MINOBJECTS,
         unsigned int MAX_DEPTH = TREE_MAXDEPTH,
         unsigned int USELESS_SPLIT_THRESHOLD = TREE_USELESS_SPLIT_THRESHOLD>
CUDA_DEVICE_CALL inline void BuildTree(Node<ElemType, Container>* tree, const ObjectAABBTesterFunction<ElemType>& ObjectBoxTester)
{
	using NodeType = Node<ElemType, Container>;

	if (!tree->IsEmpty())
	{
		Container<NodeType*> build_queue;
		build_queue.push_back(tree);

		while (!build_queue.empty())
		{
			NodeType* current_node = build_queue.back();
			build_queue.pop_back();

			if (current_node->GetSize() > MIN_OBJECTS && current_node->depth < MAX_DEPTH)
			{
				current_node->right = new NodeType(current_node->depth + 1);
				current_node->left = new NodeType(current_node->depth + 1);

				if (SplitAndGetDuplicationPercentage(*current_node, *current_node->right, *current_node->left, ObjectBoxTester) < USELESS_SPLIT_THRESHOLD)
				{
					build_queue.push_back(current_node->right);
					build_queue.push_back(current_node->left);

					current_node->elems.clear();
					current_node->elems.shrink_to_fit();
				}
				else
				{
					delete current_node->right;
					delete current_node->left;

					current_node->right = nullptr;
					current_node->left = nullptr;
				}
			}
		}
	}
}

template<typename ElemType, template<class...> class Container = std::vector, unsigned int STACK_SIZE = TREE_MAXDEPTH + 1>
CUDA_DEVICE_CALL bool IntersectsWithTree(const Node<ElemType, Container>* tree, const Ray& ray, HitData& inout_intersection, const ObjectsRayTesterFunction<ElemType>& ObjectTester)
{
	using NodeType = Node<ElemType, Container>;

	FixedSizeStack<const NodeType*, STACK_SIZE> traversal_helper;

	bool hit_something = false;
	
	const NodeType* current = tree;
	while (current || !traversal_helper.IsEmpty())
	{
		while (current && IntersectsWithBoundingBox(current->aabb, ray))
		{
			traversal_helper.Push(current);
			current = current->left;
		}

		if (!traversal_helper.IsEmpty())
		{
			current = traversal_helper.Pop();

			if (!current->IsEmpty() && ObjectTester(&current->elems[0], (unsigned int)current->GetSize(), ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		current = current->right;
	}
	
	return hit_something;
}

}
