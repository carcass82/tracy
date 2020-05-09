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
#include "collision.h"

namespace accel
{
constexpr unsigned int TREE_MAXDEPTH = 32;                 // all nodes with a depth of 32 become a leaf
constexpr unsigned int TREE_MINOBJECTS = 16;               // if after a split a node contains 16 objects or less it becomes a leaf
constexpr unsigned int TREE_USELESS_SPLIT_THRESHOLD = 200; // if a split produces 200% of initial nodes we say it's useless and nodes become a leaf


template <typename T, int FIXED_SIZE>
class FixedSizeStack
{
public:
	CUDA_DEVICE_CALL void Push(T item)    { array_[++head_] = item; }
	CUDA_DEVICE_CALL T    Pop()           { return array_[head_--]; }
	CUDA_DEVICE_CALL bool IsEmpty() const { return head_ == -1; }
	CUDA_DEVICE_CALL bool IsFull() const  { return head_ + 1 == FIXED_SIZE; }

private:
	int head_ = -1;
	T array_[FIXED_SIZE];
};

enum Child { Left, Right, Count };

template <typename T, template<class...> class Container = std::vector>
class Node
{
public:
	Node(const BBox& in_aabb, unsigned int in_depth = 0)
		: aabb(in_aabb)
		, depth(in_depth)
	{}

	Node(unsigned int in_depth = 0)
		: Node({ FLT_MAX, -FLT_MAX }, in_depth)
	{}

	bool IsEmpty() const                       { return elems.empty(); }
	unsigned int GetSize() const               { return End() - Begin(); }
	unsigned int Begin() const                 { return 0; }
	unsigned int End() const                   { return (unsigned int)elems.size(); }
	const BBox& GetAABB() const                { return aabb; }
	void SetAABB(const BBox& value)            { aabb = value; }
	const Node* GetChild(Child child) const    { return children[child]; }
	Node* GetChild(Child child)                { return children[child]; }
	void SetChild(Child child, Node* value)    { children[child] = value; }
	const T& GetElement(unsigned int i) const  { return elems[i]; }
	T& GetElement(unsigned int i)              { return elems[i]; }
	Container<T>& GetElements()                { return elems; }
	const Container<T>& GetElements() const    { return elems; }
	const T* GetData() const                   { return &elems[0]; }
	void ClearChild(Child child)               { delete children[child]; children[child] = nullptr; }
	void ClearChildren()                       { ClearChild(Child::Right); ClearChild(Child::Left); }
	void ClearElems()                          { Container<T>().swap(elems); }
	unsigned int GetDepth() const              { return depth; }

private:
	Node* children[Child::Count] = {};
	BBox aabb = { FLT_MAX, -FLT_MAX };
	Container<T> elems;
	unsigned int depth = 0;
};

template <typename NodeRoot, typename T>
struct FlatNode
{
	CUDA_CALL bool IsEmpty() const                        { return first == last; }
	CUDA_CALL const T* GetData() const                    { return &root->elements_[0]; }
	CUDA_CALL unsigned int Begin() const                  { return first; }
	CUDA_CALL unsigned int End() const                    { return last; }
	CUDA_CALL const BBox& GetAABB() const                 { return aabb; }
	CUDA_CALL const FlatNode* GetChild(Child child) const { return root->GetChild(children[child]); }
	CUDA_CALL FlatNode* GetChild(Child child)             { return root->GetChild(children[child]); }


	FlatNode(const NodeRoot* in_root = nullptr)
		: first(0)
		, last(0)
		, children{ UINT32_MAX, UINT32_MAX }
		, root(in_root)
	{}

	BBox aabb;
	unsigned int first;
	unsigned int last;
	unsigned int children[Child::Count];
	const NodeRoot* root;
};

template<typename T>
struct FlatTree
{
	CUDA_CALL const FlatNode<FlatTree, T>* GetChild(unsigned int idx) const { return (idx < nodes_num_) ? &nodes_[idx] : nullptr; }

	unsigned int nodes_num_;
	FlatNode<FlatTree, T>* nodes_;

	unsigned int elements_num_;
	T* elements_;
};

template <typename T>
using ObjectAABBTesterFunction = function<bool(const T&, const BBox&)>;

template <typename T>
using ObjectsRayTesterFunction = function<bool(const T* elems, unsigned int first, unsigned int last, const Ray&, HitData&)>;

template<typename T, template<class...> class Container = std::vector, class Predicate>
void CopyIf(const Container<T>& src, Container<T>& dest, Predicate Pred)
{
	for (auto& src_object : src)
	{
		if (Pred(src_object))
		{
			dest.push_back(src_object);
		}
	}
}

template <typename T, class NodeType>
inline unsigned int SplitAndGetDuplicationPercentage(const NodeType& current_node, NodeType& right_node, NodeType& left_node, const ObjectAABBTesterFunction<T>& ObjectBoxTester)
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

	constexpr float TRAVERSAL_COST = 1.f;
	constexpr float TRIINTERSECTION_COST = 2.f;

	float no_split_cost = TRAVERSAL_COST + TRIINTERSECTION_COST * current_node.GetSize();
	
	float best_cost = FLT_MAX;
	unsigned int best_axis = current_node.GetDepth() % 3;
	unsigned int best_split = 5;

	for (int axis = 0; axis < 3; ++axis)
	{
		for (int i = 1; i < 10; ++i)
		{
			unsigned int right_count = 0;
			unsigned int left_count = 0;

			float left_factor = i / 10.f;
			BBox left_bbox(current_node.GetAABB());
			left_bbox.maxbound[axis] -= ((current_node.GetAABB().GetSize()[axis] * left_factor) + ROUND);

			float right_factor = (10 - i) / 10.f;
			BBox right_bbox(current_node.GetAABB());
			right_bbox.minbound[axis] += ((current_node.GetAABB().GetSize()[axis] * right_factor) + ROUND);

			for (const T& object : current_node.GetElements())
			{
				right_count += (ObjectBoxTester(object, right_bbox))? 1 : 0;
				left_count  += (ObjectBoxTester(object, left_bbox ))? 1 : 0;
			}

			float cost = TRAVERSAL_COST * 2 + left_factor * (TRIINTERSECTION_COST * right_count) + right_factor * (TRIINTERSECTION_COST * left_count);
			if (cost < best_cost && cost < no_split_cost)
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


template <typename ElemType,
          template<class...> class Container = std::vector,
          typename NodeType = Node<ElemType, Container>,
          unsigned int MIN_OBJECTS = TREE_MINOBJECTS,
          unsigned int MAX_DEPTH = TREE_MAXDEPTH,
          unsigned int USELESS_SPLIT_THRESHOLD = TREE_USELESS_SPLIT_THRESHOLD>
inline void BuildTree(NodeType* tree, const ObjectAABBTesterFunction<ElemType>& ObjectBoxTester)
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

template<typename NodeElementType,
         typename FlatNodeElementType = NodeElementType,
         template<class...> class Container = std::vector,
         typename NodeType = Node<NodeElementType, Container>,
         typename FlatTreeType = FlatTree<FlatNodeElementType>,
         typename FlatNodeType = FlatNode<FlatTreeType, FlatNodeElementType>>
inline void FlattenTree(NodeType& in_SrcTree, FlatTreeType& out_FlatTree)
{
	Container<FlatNodeType> nodes;
	Container<FlatNodeElementType> elements;

	using accel::Child;
	using BuildIdx = std::pair<unsigned int /* array_pos */, NodeType* /* srctree_node */>;

	Container<BuildIdx> build_queue;

	nodes.push_back(FlatNodeType());
	build_queue.push_back(BuildIdx(0, &in_SrcTree));

	while (!build_queue.empty())
	{
		auto current_node = build_queue.back();
		build_queue.pop_back();

		if (current_node.second)
		{
			nodes[current_node.first].aabb = current_node.second->GetAABB();

			if (current_node.second->IsEmpty())
			{
				unsigned int right_child = (unsigned int)nodes.size();
				nodes[current_node.first].children[Child::Right] = right_child;
				nodes.push_back(FlatNodeType());
				build_queue.push_back(BuildIdx(right_child, current_node.second->GetChild(Child::Right)));

				unsigned int left_child = (unsigned int)nodes.size();
				nodes[current_node.first].children[Child::Left] = left_child;
				nodes.push_back(FlatNodeType());
				build_queue.push_back(BuildIdx(left_child, current_node.second->GetChild(Child::Left)));
			}
			else
			{
				nodes[current_node.first].first = (unsigned int)elements.size();
				elements.insert(elements.end(), current_node.second->GetElements().begin(), current_node.second->GetElements().end());
				nodes[current_node.first].last = (unsigned int)elements.size();
			}
		}
	}

	out_FlatTree.nodes_num_ = (unsigned int)nodes.size();
	out_FlatTree.nodes_ = new FlatNodeType[nodes.size()];
	memcpy(out_FlatTree.nodes_, &nodes[0], nodes.size() * sizeof(FlatNodeType));

	out_FlatTree.elements_num_ = (unsigned int)elements.size();
	out_FlatTree.elements_ = new FlatNodeElementType[elements.size()];
	memcpy(out_FlatTree.elements_, &elements[0], elements.size() * sizeof(FlatNodeElementType));
}

template <typename ElemType, template<class...> class Container = std::vector, typename NodeType = Node<ElemType, Container>, unsigned int STACK_SIZE = TREE_MAXDEPTH + 1>
CUDA_DEVICE_CALL bool IntersectsWithTree(const NodeType* tree, const Ray& ray, HitData& inout_intersection, const ObjectsRayTesterFunction<ElemType>& ObjectTester)
{
	FixedSizeStack<const NodeType*, STACK_SIZE> traversal_helper;

	bool hit_something = false;
	
	const NodeType* current = tree;
	while (current || !traversal_helper.IsEmpty())
	{
		while (current && collision::RayAABB(ray, current->GetAABB()))
		{
			traversal_helper.Push(current);
			current = current->GetChild(Child::Left);
		}

		if (!traversal_helper.IsEmpty())
		{
			current = traversal_helper.Pop();

			if (!current->IsEmpty() && ObjectTester(current->GetData(), current->Begin(), current->End(), ray, inout_intersection))
			{
				hit_something = true;
			}
		}

		current = current->GetChild(Child::Right);
	}
	
	return hit_something;
}

}
