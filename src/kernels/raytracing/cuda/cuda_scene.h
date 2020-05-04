/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <curand_kernel.h>

#include "cuda_log.h"
#include "camera.h"

#if !defined(CUDA_PREFERRED_DEVICE)
 #define CUDA_PREFERRED_DEVICE 0
#endif

#if USE_KDTREE
#include "kdtree.h"
using accel::Child;

#include "triangle.h"
#include "container.h"

struct TriInfo
{
	CUDA_DEVICE_CALL constexpr TriInfo(uint32_t mesh_idx = 0, uint32_t triangle_idx = 0)
		: packed((mesh_idx << 24) | triangle_idx)
	{}

	CUDA_DEVICE_CALL constexpr uint32_t GetMeshId() const { return packed >> 24; }
	CUDA_DEVICE_CALL constexpr uint32_t GetTriangleId() const { return packed & 0xffffff; }

	uint32_t packed;
};

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
    CUDA_DEVICE_CALL const CustomNode<CUDATree, TriInfo>* GetChild(unsigned int idx) const
	{
		return (idx < nodes_num_) ? &nodes_[idx] : nullptr;
	}

    unsigned int nodes_num_;
	CustomNode<CUDATree, TriInfo>* nodes_;

    unsigned int triangles_num_;
	TriInfo* triangles_;
};
#endif

struct CUDAScene
{
    int width;
    int height;

    vec4* d_output_;

	Mesh* d_objects_;
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

	static Material* ConvertMaterial(const Material* in_host_material)
	{
		if (device_materials_.count(in_host_material) == 0)
		{
			Material* d_material;
			CUDAAssert(cudaMalloc(&d_material, sizeof(Material)));
			CUDAAssert(cudaMemcpy(d_material, in_host_material, sizeof(Material), cudaMemcpyHostToDevice));

			device_materials_[in_host_material] = d_material;
		}

		return device_materials_[in_host_material];
	}

	static unordered_map<const Material*, Material*> device_materials_;
};
