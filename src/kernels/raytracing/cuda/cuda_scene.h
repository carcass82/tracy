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

// max gpu supported
constexpr int MAX_GPU = 32;

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

using CUDATree = accel::FlatTree<TriInfo>;
using CUDANode = accel::FlatNode<TriInfo, CUDATree>;

#endif

struct CUDAScene
{
    int width;
    int height;

	int num_gpus;

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
