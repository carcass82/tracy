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
//////////////////////////////////////////////////////////////////////////////
// default tree configuration                                               //
//////////////////////////////////////////////////////////////////////////////
constexpr uint32_t TREE_MAXDEPTH = 32;                 // all nodes with a depth of 32 become a leaf
constexpr uint32_t TREE_MINOBJECTS = 16;               // if after a split a node contains 16 objects or less it becomes a leaf
constexpr uint32_t TREE_USELESS_SPLIT_THRESHOLD = 200; // if a split produces 200% of initial nodes we say it's useless and nodes become a leaf

// signature for needed Element vs AABB tester function
template <typename T>
using ObjectAABBTesterFunction = function<bool(const T&, const BBox&)>;

// signature for needed Element vs Ray tester function
template <typename T>
using ObjectsRayTesterFunction = function<bool(const T* elems, uint32_t first, uint32_t last, const Ray&, collision::HitData&)>;


//////////////////////////////////////////////////////////////////////////////
// Tree data structures                                                     //
//////////////////////////////////////////////////////////////////////////////

enum Child { Left, Right, Count };

//
// Basic Tree node with all needed informations
//
template <typename T, template<class...> class Container = std::vector>
class Node
{
public:
	Node(const BBox& in_aabb, uint32_t in_depth = 0)
		: aabb_(in_aabb)
		, depth_(in_depth)
	{}

	Node(uint32_t in_depth = 0)
		: Node({ FLT_MAX, -FLT_MAX }, in_depth)
	{}

	bool IsEmpty() const                       { return elems_.empty(); }
	uint32_t GetSize() const                   { return End() - Begin(); }
	uint32_t Begin() const                     { return 0u; }
	uint32_t End() const                       { return static_cast<uint32_t>(elems_.size()); }
	const BBox& GetAABB() const                { return aabb_; }
	void SetAABB(const BBox& value)            { aabb_ = value; }
	const Node* GetChild(Child child) const    { return children_[child]; }
	Node* GetChild(Child child)                { return children_[child]; }
	void SetChild(Child child, Node* value)    { children_[child] = value; }
	const T& GetElement(unsigned int i) const  { return elems_[i]; }
	T& GetElement(unsigned int i)              { return elems_[i]; }
	Container<T>& GetElements()                { return elems_; }
	const Container<T>& GetElements() const    { return elems_; }
	const T* GetData() const                   { return &elems_[0]; }
	void ClearChild(Child child)               { delete children_[child]; children_[child] = nullptr; }
	void ClearChildren()                       { ClearChild(Child::Right); ClearChild(Child::Left); }
	void ClearElems()                          { Container<T>().swap(elems_); }
	uint32_t GetDepth() const                  { return depth_; }

private:
	Node* children_[Child::Count]{};
	BBox aabb_{ FLT_MAX, -FLT_MAX };
	Container<T> elems_{};
	unsigned int depth_{ 0 };
};

//
// Sample Node containing just connection infos with other nodes but no elements
//
template <typename T, typename NodeRoot>
struct FlatNode
{
	CUDA_CALL bool IsEmpty() const                        { return first == last; }
	CUDA_CALL const T* GetData() const                    { return &root->elements[0]; }
	CUDA_CALL uint32_t Begin() const                      { return first; }
	CUDA_CALL uint32_t End() const                        { return last; }
#if USE_INTRINSICS
	CUDA_CALL __m128 GetAABBMin() const                   { return aabb_min; }
	CUDA_CALL __m128 GetAABBMax() const                   { return aabb_max; }
#else
	CUDA_CALL const BBox& GetAABB() const                 { return aabb; }
	CUDA_CALL const vec3& GetAABBMin() const              { return aabb.minbound; }
	CUDA_CALL const vec3& GetAABBMax() const              { return aabb.maxbound; }
#endif
	CUDA_CALL const FlatNode* GetChild(Child child) const { return root->GetChild(children[child]); }
	CUDA_CALL FlatNode* GetChild(Child child)             { return root->GetChild(children[child]); }

	FlatNode(const NodeRoot* in_root = nullptr)
		: first{ 0 }
		, last{ 0 }
		, children{ UINT32_MAX, UINT32_MAX }
		, root{ in_root }
	{}

#if USE_INTRINSICS
	__m128 aabb_min;
	__m128 aabb_max;
#else
	BBox aabb;
#endif
	uint32_t first;
	uint32_t last;
	uint32_t children[Child::Count];
	const NodeRoot* root;
};

//
// Sample tree containing both nodes and elements layed out in plain arrays
//
template <typename T>
struct FlatTree
{
	CUDA_CALL const FlatNode<T, FlatTree>* GetChild(uint32_t idx) const { return (idx < nodes_num)? &nodes[idx] : nullptr; }

	uint32_t nodes_num;
	FlatNode<T, FlatTree>* nodes;

	uint32_t elements_num;
	T* elements;
};


//////////////////////////////////////////////////////////////////////////////
// Tree building functions                                                  //
//////////////////////////////////////////////////////////////////////////////

//
// naive CopyIf
// because std functions may not be available (e.g. cuda/gpu)
//
template<typename T, template<class...> class Container = std::vector, class Predicate>
void CopyIf(const Container<T>& src, Container<T>& dest, Predicate Pred)
{
	for (const auto& src_object : src)
	{
		if (Pred(src_object))
		{
			dest.push_back(src_object);
		}
	}
}

//
// splits current node in right and left node using Object vs AABB function
//
template <typename T, class NodeType>
inline unsigned int SplitAndGetDuplicationPercentage(const NodeType& current_node, NodeType& right_node, NodeType& left_node, const ObjectAABBTesterFunction<T>& ObjectBoxTester)
{
	constexpr float ROUND = 1.e-4f;

#if defined(DISABLE_SAH)

	uint32_t axis = current_node.GetDepth() % 3;
	float split = (current_node.GetAABB().GetSize()[axis] / 2.f) + ROUND;

	BBox left_bbox(current_node.GetAABB());
	left_bbox.maxbound[axis] -= split;
	left_node.SetAABB(left_bbox);

	BBox right_bbox(current_node.GetAABB());
	right_bbox.minbound[axis] += split;
	right_node.SetAABB(right_bbox);

#else

	static constexpr float TRAVERSAL_COST = 1.f;
	static constexpr float TRIINTERSECTION_COST = 2.f;

	float no_split_cost = TRAVERSAL_COST + TRIINTERSECTION_COST * current_node.GetSize();
	
	float best_cost = FLT_MAX;
	uint32_t best_axis = current_node.GetDepth() % 3;
	uint32_t best_split = 5;

	for (int axis = 0; axis < 3; ++axis)
	{
		for (int i = 1; i < 10; ++i)
		{
			uint32_t right_count = 0;
			uint32_t left_count = 0;

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

//
// BuildTree:
// starting from tree object containing all nodes and AABB generates a full kd-tree based on policy parameters
//
template <typename ElemType,
          template<class...> class Container = std::vector,
          typename NodeType = Node<ElemType, Container>,
          uint32_t MIN_OBJECTS = TREE_MINOBJECTS,
          uint32_t MAX_DEPTH = TREE_MAXDEPTH,
          uint32_t USELESS_SPLIT_THRESHOLD = TREE_USELESS_SPLIT_THRESHOLD>
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

				if (SplitAndGetDuplicationPercentage(*current_node,
					                                 *current_node->GetChild(Child::Right),
					                                 *current_node->GetChild(Child::Left),
					                                 ObjectBoxTester) < USELESS_SPLIT_THRESHOLD)
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

//
// takes a full kdtree (like the one generated by BuildTree) and lay out all its
// nodes to a FlatTree array-based structure
//
template<typename NodeElementType,
         typename FlatNodeElementType = NodeElementType,
         template<class...> class Container = std::vector,
         typename NodeType = Node<NodeElementType, Container>,
         typename FlatTreeType = FlatTree<FlatNodeElementType>,
         typename FlatNodeType = FlatNode<FlatNodeElementType, FlatTreeType>>
inline void FlattenTree(NodeType& in_SrcTree, FlatTreeType& out_FlatTree)
{
	Container<FlatNodeType> nodes;
	Container<FlatNodeElementType> elements;

	using accel::Child;
	using BuildIdx = std::pair<uint32_t /* array_pos */, NodeType* /* srctree_node */>;

	Container<BuildIdx> build_queue;

	nodes.push_back(FlatNodeType());
	build_queue.push_back(BuildIdx(0, &in_SrcTree));

	while (!build_queue.empty())
	{
		auto current_node = build_queue.back();
		build_queue.pop_back();

		nodes[current_node.first].root = &out_FlatTree;

		if (current_node.second)
		{
#if USE_INTRINSICS
			vec3 tmp = current_node.second->GetAABB().minbound;
			nodes[current_node.first].aabb_min = _mm_set_ps(tmp.z, tmp.z, tmp.y, tmp.x);

			tmp = current_node.second->GetAABB().maxbound;
			nodes[current_node.first].aabb_max = _mm_set_ps(tmp.z, tmp.z, tmp.y, tmp.x);
#else
			nodes[current_node.first].aabb = current_node.second->GetAABB();
#endif

			if (current_node.second->IsEmpty())
			{
				uint32_t right_child = static_cast<uint32_t>(nodes.size());
				nodes[current_node.first].children[Child::Right] = right_child;
				nodes.push_back(FlatNodeType());
				build_queue.push_back(BuildIdx(right_child, current_node.second->GetChild(Child::Right)));

				uint32_t left_child = static_cast<uint32_t>(nodes.size());
				nodes[current_node.first].children[Child::Left] = left_child;
				nodes.push_back(FlatNodeType());
				build_queue.push_back(BuildIdx(left_child, current_node.second->GetChild(Child::Left)));
			}
			else
			{
				nodes[current_node.first].first = static_cast<uint32_t>(elements.size());
				elements.insert(elements.end(), current_node.second->GetElements().begin(), current_node.second->GetElements().end());
				nodes[current_node.first].last = static_cast<uint32_t>(elements.size());
			}
		}
	}

	out_FlatTree.nodes_num = static_cast<uint32_t>(nodes.size());
	out_FlatTree.nodes = new FlatNodeType[nodes.size()];
	memcpy(out_FlatTree.nodes, &nodes[0], nodes.size() * sizeof(FlatNodeType));

	out_FlatTree.elements_num = static_cast<uint32_t>(elements.size());
	out_FlatTree.elements = new FlatNodeElementType[elements.size()];
	memcpy(out_FlatTree.elements, &elements[0], elements.size() * sizeof(FlatNodeElementType));
}


//////////////////////////////////////////////////////////////////////////////
// Tree traversal function                                                  //
//////////////////////////////////////////////////////////////////////////////

//
// quick and dirty stack
//
template <typename T, int32_t FIXED_SIZE>
class FixedSizeStack
{
public:
	CUDA_DEVICE_CALL void Push(T item)    { array_[++head_] = item; }
	CUDA_DEVICE_CALL T    Pop()           { return array_[head_--]; }
	CUDA_DEVICE_CALL bool IsEmpty() const { return head_ == -1; }
	CUDA_DEVICE_CALL bool IsFull() const  { return head_ + 1 == FIXED_SIZE; }
	CUDA_DEVICE_CALL void Clear()         { head_ = -1; }

private:
	int32_t head_{ -1 };
	T array_[FIXED_SIZE];
};

//
// computes intersections of input ray with elements contained in input tree
//
template <typename ElemType,
          template<class...> class Container = std::vector,
          typename NodeType = Node<ElemType, Container>,
          uint32_t STACK_SIZE = TREE_MAXDEPTH + 1>
CUDA_DEVICE_CALL bool IntersectsWithTree(const NodeType* tree, const Ray& ray, collision::HitData& inout_intersection, const ObjectsRayTesterFunction<ElemType>& ObjectTester)
{
	FixedSizeStack<const NodeType*, STACK_SIZE> traversal_helper;

#if USE_INTRINSICS
	vec3 origin = ray.GetOrigin();
	vec3 inv_direction = ray.GetDirectionInverse();

	__m128 rayO{ _mm_set_ps(origin.z, origin.z, origin.y, origin.x) };
	__m128 rayI{ _mm_set_ps(inv_direction.z, inv_direction.z, inv_direction.y, inv_direction.x) };
#else
	const vec3 rayO{ ray.GetOrigin() };
	const vec3 rayI{ ray.GetDirectionInverse() };
#endif

	float minT = inout_intersection.t;

	bool hit_something = false;
	
	const NodeType* current = tree;
	while (current || !traversal_helper.IsEmpty())
	{
		while (current && collision::RayAABB(rayO, rayI, current->GetAABBMin(), current->GetAABBMax(), minT))
		{
			traversal_helper.Push(current);
			current = current->GetChild(Child::Left);
		}

		if (!traversal_helper.IsEmpty())
		{
			current = traversal_helper.Pop();

			if (!current->IsEmpty() && ObjectTester(current->GetData(), current->Begin(), current->End(), ray, inout_intersection))
			{
				minT = inout_intersection.t;
				hit_something = true;
			}
		}

		current = current->GetChild(Child::Right);
	}
	
	return hit_something;
}

}
