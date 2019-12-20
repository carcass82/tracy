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
 #define CUDA_PREFERRED_DEVICE 1
#endif

struct CUDAScene
{
    ~CUDAScene()
    {
        CUDAAssert(cudaFree(objects_));
        CUDAAssert(cudaFree(d_camera_));
        CUDAAssert(cudaFree(d_rand_state));
        CUDAAssert(cudaFree(d_raycount));
    }

    int width;
    int height;

	CUDAMesh* objects_;
	int objectcount_;

	Camera* d_camera_;

    curandState* d_rand_state;

    int* h_raycount;
    int* d_raycount;
};
