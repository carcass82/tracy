/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <cuda_runtime.h>
#include "cuda_log.h"
#include "common.h"
#include "vertex.h"
#include "mesh.h"
#include "material.h"

struct CUDAMesh
{
public:
	__host__ CUDAMesh()
	{}

	__host__ CUDAMesh(const Mesh& cpu_mesh, const Material* d_material)
		: material_(d_material)
		, center_(cpu_mesh.GetCenter())
		, size_(cpu_mesh.GetSize())
		, aabb_(cpu_mesh.GetAABB())
	{
		vertexcount_ = cpu_mesh.GetVertexCount();
		CUDAAssert(cudaMalloc(&vertices_, vertexcount_ * sizeof(Vertex)));
		CUDAAssert(cudaMemcpy(vertices_, &cpu_mesh.GetVertices()[0], vertexcount_ * sizeof(Vertex), cudaMemcpyHostToDevice));

		indexcount_ = cpu_mesh.GetIndexCount();
		CUDAAssert(cudaMalloc(&indices_, indexcount_ * sizeof(Index)));
		CUDAAssert(cudaMemcpy(indices_, &cpu_mesh.GetIndices()[0], indexcount_* sizeof(Index), cudaMemcpyHostToDevice));
	}

	Vertex* vertices_;
	int vertexcount_;
	Index* indices_;
	int indexcount_;

	const Material* material_;
	
	vec3 center_;
	vec3 size_;
	BBox aabb_;
};
