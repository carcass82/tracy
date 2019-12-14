/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <cuda_runtime.h>
#include "log.h"
#include "common.h"
#include "vertex.h"
#include "mesh.h"
#include "material.h"

class CUDAMesh
{
public:
	__host__ CUDAMesh()
	{}

	__host__ CUDAMesh(const Mesh& cpu_mesh)
		: center_(cpu_mesh.GetCenter())
		, size_(cpu_mesh.GetSize())
		, aabb_(cpu_mesh.GetAABB())
	{
		cudaMalloc((void**)&material_, sizeof(Material));
		cudaMemcpy(material_, cpu_mesh.GetMaterial(), sizeof(Material), cudaMemcpyHostToDevice);

		vertexcount_ = cpu_mesh.GetVertexCount();
		cudaMalloc((void**)&vertices_, vertexcount_ * sizeof(Vertex));
		cudaMemcpy(vertices_, &(cpu_mesh.GetVertices())[0], vertexcount_, cudaMemcpyHostToDevice);

		indexcount_ = cpu_mesh.GetIndicesCount();
		cudaMalloc((void**)&indices_, indexcount_ * sizeof(Index));
		cudaMemcpy(indices_, &(cpu_mesh.GetIndices())[0], indexcount_, cudaMemcpyHostToDevice);
	}

	__host__ ~CUDAMesh()
	{
		cudaFree(material_);
	}
	
	CUDA_CALL int GetVertexCount() const            { return vertexcount_; }
	
	CUDA_CALL int GetTriCount() const               { return indexcount_ / 3; }
	
	CUDA_CALL int GetIndicesCount() const           { return indexcount_; }
	
	__device__ const Vertex* GetVertices() const    { return vertices_; }
	
	__device__ const Vertex& GetVertex(int i) const { return vertices_[i]; }
	
	__device__ const Index* GetIndices() const      { return indices_; }
	
	__device__ const Index& GetIndex(int i) const   { return indices_[i]; }
	
	__device__ const Material* GetMaterial() const  { return material_; }
	
	CUDA_CALL const vec3& GetCenter() const         { return center_; }
	
	CUDA_CALL const vec3& GetSize() const           { return size_; }
	
	CUDA_CALL const BBox& GetAABB() const           { return aabb_; }

private:
	Vertex* vertices_;
	int vertexcount_;
	Index* indices_;
	int indexcount_;
	
	Material* material_;

	vec3 center_;
	vec3 size_;
	BBox aabb_;
};
