/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include "camera.h"
#include "cuda_mesh.h"

struct CUDAScene
{
	CUDAMesh* objects_;
	int objectcount_;

	Camera* d_camera_;
};
