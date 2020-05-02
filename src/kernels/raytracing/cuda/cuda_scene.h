/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <curand_kernel.h>

#include "camera.h"
#include "cuda_mesh.h"

#if !defined(CUDA_PREFERRED_DEVICE)
 #define CUDA_PREFERRED_DEVICE 0
#endif

#if USE_KDTREE
#include "kdtree.h"
using accel::Child;

#include "triangle.h"
#include "container.h"


template <typename Father, typename T>
struct CustomNode
{
	CUDA_DEVICE_CALL bool IsEmpty() const                          { return first == last; }
	CUDA_DEVICE_CALL const T* GetData() const                      { return &root->triangles_[0]; }
	CUDA_DEVICE_CALL unsigned int Begin() const                    { return first; }
	CUDA_DEVICE_CALL unsigned int End() const                      { return last; }
	CUDA_DEVICE_CALL const BBox& GetAABB() const                   { return aabb; }
	CUDA_DEVICE_CALL const CustomNode* GetChild(Child child) const { return root->GetChild(children[child]); }
	CUDA_DEVICE_CALL CustomNode* GetChild(Child child)             { return root->GetChild(children[child]); }


	/* __host__ */ CustomNode(const Father* in_root = nullptr)
		: first(0), last(0), children{ UINT32_MAX, UINT32_MAX }, root(in_root)
	{}

	BBox aabb;                           // 12

	unsigned int first;                  // 4
	unsigned int last;                   // 4
	unsigned int children[Child::Count]; // 8

	const Father* root;                  // 4
};

struct CUDATree
{
    CUDA_DEVICE_CALL const CustomNode<CUDATree, Triangle>* GetChild(unsigned int idx) const
	{
		return (idx < nodes_num_) ? &nodes_[idx] : nullptr;
	}

    unsigned int nodes_num_;
	CustomNode<CUDATree, Triangle>* nodes_;

    unsigned int triangles_num_;
	Triangle* triangles_;
};
#endif

struct CUDAScene
{
    int width;
    int height;

    vec4* d_output_;

	CUDAMesh* d_objects_;
	int objectcount_;

#if USE_KDTREE
    CUDATree h_scenetree;
    CUDATree* d_scenetree;
#endif

    Material* d_sky_;

	Camera* d_camera_;

    curandState* d_rand_state;

    int h_raycount;
    int* d_raycount;

    
    int GetRayCount()
    {
        CUDAAssert(cudaMemcpy(&h_raycount, d_raycount, sizeof(int), cudaMemcpyDeviceToHost));
        return h_raycount;
    }
    
    void ResetRayCount()
    {
        CUDAAssert(cudaMemset(d_raycount, 0, sizeof(int)));
        h_raycount = 0;
    }
};
