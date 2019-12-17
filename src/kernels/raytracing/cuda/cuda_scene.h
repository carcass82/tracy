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

struct CUDAScene
{
    ~CUDAScene()
    {
        cudaFree(objects_);
        cudaFree(d_camera_);
        cudaFree(d_rand_state);
        cudaFree(d_raycount);
    }

    int width;
    int height;

	CUDAMesh* objects_;
	int objectcount_;

	Camera* d_camera_;

    curandState* d_rand_state;

    int* d_raycount;
};
