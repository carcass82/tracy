/*
 * Tracy, a simple raytracer
 * inspired by "Ray Tracing in One Weekend" minibooks
 *
 * (c) Carlo Casta, 2018
 */
#pragma once

#include <vector>
using std::vector;

#include "common.h"
#include "aabb.h"
#include "material.h"

#if defined(CPU_KERNEL)
#include "kernels/raytracing/software/cpu_specific.h"
#elif defined(CUDA_KERNEL)
#include "kernels/raytracing/cuda/cuda_specific.h"
#elif defined(OPENGL_KERNEL)
#include "kernels/rasterization/opengl/opengl_specific.h"
#elif defined(DXR_KERNEL)
#include "kernels/raytracing/dxr/dxr_specific.h"
#else
#error "at least one module should be enabled!"
#endif

class Mesh
{
public:
	Mesh() {}

	Mesh(const vector<Vertex>& in_vertices, const vector<Index>& in_indices, const Material* in_material = nullptr)
		: material_(in_material)
	{
		DEBUG_ASSERT(in_vertices.size() < UINT32_MAX && in_indices.size() < UINT32_MAX);

		vertexcount_ = static_cast<uint32_t>(in_vertices.size());
		vertices_ = new Vertex[vertexcount_];
		memcpy(vertices_, &in_vertices[0], vertexcount_ * sizeof(Vertex));

		indexcount_ = static_cast<uint32_t>(in_indices.size());
		indices_ = new Index[indexcount_];
		memcpy(indices_, &in_indices[0], indexcount_ * sizeof(Index));
	}

	~Mesh()
	{
		delete [] vertices_;
		delete [] indices_;
	}

	Mesh(const Mesh& other) = delete;

	Mesh& operator=(Mesh& other) = delete;

    Mesh(Mesh&& other) noexcept 
		: vertices_(std::exchange(other.vertices_, nullptr))
		, vertexcount_(std::exchange(other.vertexcount_, 0))
		, indices_(std::exchange(other.indices_, nullptr))
		, indexcount_(std::exchange(other.indexcount_, 0))
		, material_(std::exchange(other.material_, nullptr))
		, aabb_(std::move(other.aabb_))
    {
    }

    Mesh& operator=(Mesh&& other) noexcept
    {
        vertices_ = std::exchange(other.vertices_, nullptr);
		vertexcount_ = std::exchange(other.vertexcount_, 0);
		indices_ = std::exchange(other.indices_, nullptr);
		indexcount_ = std::exchange(other.indexcount_, 0);
		material_ = std::exchange(other.material_, nullptr);
		aabb_ = std::move(other.aabb_);
        return *this;
    }

	Mesh& ComputeNormals();
	
	template<typename VertexType = Vertex, bool enabled = VertexType::VertexHasTangents>
	typename std::enable_if<enabled, Mesh&>::type ComputeTangentsAndBitangents();

	template<typename VertexType = Vertex, bool enabled = VertexType::VertexHasTangents>
	typename std::enable_if<!enabled, Mesh&>::type ComputeTangentsAndBitangents();

	Mesh& ComputeBoundingBox();

	CUDA_CALL uint32_t GetVertexCount() const               { return vertexcount_; }

	CUDA_CALL uint32_t GetTriCount() const                  { return indexcount_ / 3; }

	CUDA_CALL uint32_t GetIndexCount() const                { return indexcount_; }

	CUDA_CALL const Vertex* GetVertices() const             { return vertices_; }

	CUDA_CALL Vertex& GetVertex(int i) const                { return vertices_[i]; }

	CUDA_CALL const Index* GetIndices() const               { return indices_; }

	CUDA_CALL Index& GetIndex(int i) const                  { return indices_[i]; }

	CUDA_CALL void SetMaterial(const Material* in_material) { material_ = in_material; }

	CUDA_CALL const Material* GetMaterial() const           { return material_; }

	CUDA_CALL const BBox& GetAABB() const                   { return aabb_; }

	CUDA_CALL void SetAABB(const BBox& in_box)              { aabb_ = in_box; }
	

private:
	Vertex* vertices_{};
	uint32_t vertexcount_{};
	Index* indices_{};
	uint32_t indexcount_{};
	const Material* material_{};
	BBox aabb_{};
};


inline Mesh& Mesh::ComputeNormals()
{
	for (uint32_t i = 0; i < indexcount_; i += 3)
	{
		Vertex& v1 = vertices_[indices_[i + 0]];
		Vertex& v2 = vertices_[indices_[i + 1]];
		Vertex& v3 = vertices_[indices_[i + 2]];

		vec3 normal = normalize(cross(v2.pos - v1.pos, v3.pos - v1.pos));
		v1.normal = v2.normal = v3.normal = normal;
	}

	return *this;
}

inline Mesh& Mesh::ComputeBoundingBox()
{
	aabb_ = { FLT_MAX, -FLT_MAX };
	for (uint32_t i = 0; i < vertexcount_; ++i)
	{
		aabb_.minbound = pmin(aabb_.minbound, vertices_[i].pos);
		aabb_.maxbound = pmax(aabb_.maxbound, vertices_[i].pos);
	}

	return *this;
}

template<typename VertexType, bool enabled>
inline typename std::enable_if<enabled, Mesh&>::type Mesh::ComputeTangentsAndBitangents()
{
	// Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh"
	// http://www.terathon.com/code/tangent.html

	for (uint32_t i = 0; i < indexcount_; i += 3)
	{
		VertexType& v1 = vertices_[indices_[i + 0]];
		VertexType& v2 = vertices_[indices_[i + 1]];
		VertexType& v3 = vertices_[indices_[i + 2]];

		// Edges of the triangle : position delta
		vec3 delta_pos1 = v2.pos - v1.pos;
		vec3 delta_pos2 = v3.pos - v1.pos;

		// UV delta
		vec2 delta_uv1 = v2.uv0 - v1.uv0;
		vec2 delta_uv2 = v3.uv0 - v1.uv0;

		float r = rcp(delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
		vec3 tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
		vec3 bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

		v1.tangent += v2.tangent = v3.tangent = tangent;
		v1.bitangent += v2.bitangent = v3.bitangent = bitangent;
	}

	for (uint32_t i = 0; i < vertexcount_; ++i)
	{
		VertexType& v = vertices_[i];

		// orthonormalize tangent
		v.tangent = normalize(v.tangent - v.normal * dot(v.normal, v.tangent));

		// ensure coherent handedness
		float sign = (dot(cross(v.normal, v.tangent), v.bitangent) < 0.0f) ? -1.0f : 1.0f;
		v.bitangent = sign * cross(v.normal, v.tangent);
	}

	return *this;
}

template<typename VertexType, bool enabled>
inline typename std::enable_if<!enabled, Mesh&>::type Mesh::ComputeTangentsAndBitangents()
{
	return *this;
}
