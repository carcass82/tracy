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

struct CUDAScene
{
    int width;
    int height;

    vec4* d_output_;

	CUDAMesh* d_objects_;
	int objectcount_;

    Material* d_sky_;

	Camera* d_camera_;

    curandState* d_rand_state;

    int h_raycount;
    int* d_raycount;
};
