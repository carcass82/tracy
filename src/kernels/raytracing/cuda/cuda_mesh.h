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

struct CUDAVertex
{
	CUDAVertex() {}

	CUDAVertex(const vec3& in_pos, const vec3& in_normal, const vec2& in_uv0)
		: pos{ in_pos }
		, normal{ in_normal }
		, uv0{ in_uv0 }
	{}

	vec3 pos;
	vec3 normal;
	vec2 uv0;
};

struct CUDAMesh
{
public:
	__host__ CUDAMesh()
	{}

	__host__ CUDAMesh(const Mesh& cpu_mesh)
		: center_(cpu_mesh.GetCenter())
		, size_(cpu_mesh.GetSize())
		, aabb_(cpu_mesh.GetAABB())
	{
		vertexcount_ = cpu_mesh.GetVertexCount();

		vector<CUDAVertex> vertices_helper;
		vertices_helper.reserve(vertexcount_);
		for (const Vertex& vertex : cpu_mesh.GetVertices())
		{
			vertices_helper.emplace_back(vertex.pos, vertex.normal, vertex.uv0);
		}
		CUDAAssert(cudaMalloc(&vertices_, vertexcount_ * sizeof(CUDAVertex)));
		CUDAAssert(cudaMemcpy(vertices_, &vertices_helper[0], vertexcount_ * sizeof(CUDAVertex), cudaMemcpyHostToDevice));

		indexcount_ = cpu_mesh.GetIndicesCount();
		CUDAAssert(cudaMalloc(&indices_, indexcount_ * sizeof(Index)));
		CUDAAssert(cudaMemcpy(indices_, &cpu_mesh.GetIndices()[0], indexcount_* sizeof(Index), cudaMemcpyHostToDevice));

		CUDAAssert(cudaMalloc(&material_, sizeof(Material)));
		CUDAAssert(cudaMemcpy(material_, cpu_mesh.GetMaterial(), sizeof(Material), cudaMemcpyHostToDevice));
	}

	__host__ ~CUDAMesh()
	{
		CUDAAssert(cudaFree(material_));
		CUDAAssert(cudaFree(indices_));
		CUDAAssert(cudaFree(vertices_));
	}
	
	//
	//
	//

	CUDAVertex* vertices_;
	int vertexcount_;
	Index* indices_;
	int indexcount_;

	Material* material_;
	
	vec3 center_;
	vec3 size_;
	BBox aabb_;
};
