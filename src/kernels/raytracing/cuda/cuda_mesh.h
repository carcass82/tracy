/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once
#include <unordered_map>
using std::unordered_map;

#include <cuda_runtime.h>
#include "cuda_log.h"
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

struct CUDAMaterial
{
	static Material* Convert(const Material* cpu_material)
	{
		if (host_to_device_.count(cpu_material) == 0)
		{
			Material* d_material;
			CUDAAssert(cudaMalloc(&d_material, sizeof(Material)));
			CUDAAssert(cudaMemcpy(d_material, cpu_material, sizeof(Material), cudaMemcpyHostToDevice));

			host_to_device_[cpu_material] = d_material;
		}

		return host_to_device_[cpu_material];
	}

	static unordered_map<const Material*, Material*> host_to_device_;
};

struct CUDAMesh
{
public:
	__host__ CUDAMesh()
	{}

	__host__ CUDAMesh(const Mesh& cpu_mesh, const Material* d_material)
		: center_(cpu_mesh.GetCenter())
		, size_(cpu_mesh.GetSize())
		, aabb_(cpu_mesh.GetAABB())
		, material_(d_material)
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

		indexcount_ = cpu_mesh.GetIndexCount();
		CUDAAssert(cudaMalloc(&indices_, indexcount_ * sizeof(Index)));
		CUDAAssert(cudaMemcpy(indices_, &cpu_mesh.GetIndices()[0], indexcount_* sizeof(Index), cudaMemcpyHostToDevice));
	}

	//
	//
	//

	CUDAVertex* vertices_;
	int vertexcount_;
	Index* indices_;
	int indexcount_;

	const Material* material_;
	
	vec3 center_;
	vec3 size_;
	BBox aabb_;
};
