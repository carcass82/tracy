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

enum Child { Left, Right, Count };

template <typename T, template<class...> class Container = std::vector>
class Node
{
public:
	CUDA_DEVICE_CALL Node(const BBox& in_aabb, unsigned int in_depth = 0)
		: aabb(in_aabb)
		, depth(in_depth)
	{}

	CUDA_DEVICE_CALL Node(unsigned int in_depth = 0)
		: Node({ FLT_MAX, -FLT_MAX }, in_depth)
	{}

	CUDA_DEVICE_CALL bool IsEmpty() const                      { return elems.empty();              }
	CUDA_DEVICE_CALL unsigned int GetSize() const              { return (unsigned int)elems.size(); }
	CUDA_DEVICE_CALL const BBox& GetAABB() const               { return aabb; }
	CUDA_DEVICE_CALL void SetAABB(const BBox& value)           { aabb = value; }
	CUDA_DEVICE_CALL const Node* GetChild(Child child) const   { return children[child]; }
	CUDA_DEVICE_CALL Node* GetChild(Child child)               { return children[child]; }
	CUDA_DEVICE_CALL void SetChild(Child child, Node* value)   { children[child] = value; }
	CUDA_DEVICE_CALL const T& GetElement(unsigned int i) const { return elems[i]; }
	CUDA_DEVICE_CALL T& GetElement(unsigned int i)             { return elems[i]; }
	CUDA_DEVICE_CALL Container<T>& GetElements()               { return elems; }
	CUDA_DEVICE_CALL const Container<T>& GetElements() const   { return elems; }
	CUDA_DEVICE_CALL void ClearChild(Child child)              { delete children[child]; children[child] = nullptr; }
	CUDA_DEVICE_CALL void ClearChildren()                      { ClearChild(Child::Right); ClearChild(Child::Left); }
	CUDA_DEVICE_CALL void ClearElems()                         { Container<T>().swap(elems); }
	CUDA_DEVICE_CALL unsigned int GetDepth() const             { return depth; }

private:
	Node* children[Child::Count] = {};
	BBox aabb = { FLT_MAX, -FLT_MAX };
	Container<T> elems;
	unsigned int depth = 0;
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

#if defined(DISABLE_SAH)

	unsigned int axis = current_node.GetDepth() % 3;
	float split = (current_node.GetAABB().GetSize()[axis] / 2.f) + ROUND;

	BBox left_bbox(current_node.GetAABB());
	left_bbox.maxbound[axis] -= split;
	left_node.SetAABB(left_bbox);

	BBox right_bbox(current_node.GetAABB());
	right_bbox.minbound[axis] += split;
	right_node.SetAABB(right_bbox);

#else

	float best_cost = FLT_MAX;
	unsigned int best_axis = current_node.GetDepth() % 3;
	unsigned int best_split = 5;

	//for (int axis = 0; axis < 3; ++axis)
	int axis = best_axis;
	{
		for (int i = 1; i < 10; ++i)
		{
			unsigned int right_count = 0;
			unsigned int left_count = 0;

			vec3 split_unit = current_node.GetAABB().GetSize() / 10.f;

			BBox left_bbox(current_node.GetAABB());
			left_bbox.maxbound[axis] -= ((split_unit[axis] * i) + ROUND);

			BBox right_bbox(current_node.GetAABB());
			right_bbox.minbound[axis] += (split_unit[axis] * (10 - i)) + ROUND;

			for (const T& object : current_node.GetElements())
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

	float split_max = (current_node.GetAABB().GetSize()[best_axis] / 10.f * best_split) + ROUND;
	float split_min = (current_node.GetAABB().GetSize()[best_axis] / 10.f * (10 - best_split)) + ROUND;

	BBox left_bbox(current_node.GetAABB());
	left_bbox.maxbound[best_axis] -= split_min;
	left_node.SetAABB(left_bbox);

	BBox right_bbox(current_node.GetAABB());
	right_bbox.minbound[best_axis] += split_max;
	right_node.SetAABB(right_bbox);

#endif

	CopyIf(current_node.GetElements(), right_node.GetElements(), [&](const T& elem) { return ObjectBoxTester(elem, right_node.GetAABB()); });
	CopyIf(current_node.GetElements(), left_node.GetElements(),  [&](const T& elem) { return ObjectBoxTester(elem, left_node.GetAABB()); });

	return (right_node.GetSize() + left_node.GetSize()) * 100 / current_node.GetSize();
}


template<typename ElemType,
         template<class...> class Container = std::vector,
         typename NodeType = Node<ElemType, Container>,
         unsigned int MIN_OBJECTS = TREE_MINOBJECTS,
         unsigned int MAX_DEPTH = TREE_MAXDEPTH,
         unsigned int USELESS_SPLIT_THRESHOLD = TREE_USELESS_SPLIT_THRESHOLD>
CUDA_DEVICE_CALL inline void BuildTree(NodeType* tree, const ObjectAABBTesterFunction<ElemType>& ObjectBoxTester)
{
	if (!tree->IsEmpty())
	{
		Container<NodeType*> build_queue;
		build_queue.push_back(tree);

		while (!build_queue.empty())
		{
			NodeType* current_node = build_queue.back();
			build_queue.pop_back();

			if (current_node->GetSize() > MIN_OBJECTS && current_node->GetDepth() < MAX_DEPTH)
			{
				current_node->SetChild(Child::Right, new NodeType(current_node->GetDepth() + 1));
				current_node->SetChild(Child::Left, new NodeType(current_node->GetDepth() + 1));

				if (SplitAndGetDuplicationPercentage(*current_node, *current_node->GetChild(Child::Right), *current_node->GetChild(Child::Left), ObjectBoxTester) < USELESS_SPLIT_THRESHOLD)
				{
					build_queue.push_back(current_node->GetChild(Child::Right));
					build_queue.push_back(current_node->GetChild(Child::Left));
					current_node->ClearElems();
				}
				else
				{
					current_node->ClearChildren();
				}
			}
		}
	}
}

template<typename ElemType, template<class...> class Container = std::vector, typename NodeType = Node<ElemType, Container>, unsigned int STACK_SIZE = TREE_MAXDEPTH + 1>
CUDA_DEVICE_CALL bool IntersectsWithTree(const NodeType* tree, const Ray& ray, HitData& inout_intersection, const ObjectsRayTesterFunction<ElemType>& ObjectTester)
{
	FixedSizeStack<const NodeType*, STACK_SIZE> traversal_helper;

	bool hit_something = false;
	
	const NodeType* current = tree;
	while (current || !traversal_helper.IsEmpty())
	{
		while (current && IntersectsWithBoundingBox(current->GetAABB(), ray))
		{
			traversal_helper.Push(current);
			current = current->GetChild(Child::Left);
		}

		if (!traversal_helper.IsEmpty())
		{
			current = traversal_helper.Pop();

			if (!current->IsEmpty() && ObjectTester(&current->GetElement(0), current->GetSize(), ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		current = current->GetChild(Child::Right);
	}
	
	return hit_something;
}

}
